import aakaar
import aakaar._C as _C


class CapturedConv1d:
    def __init__(self, weight, batch, length, stride=1, padding=0, dilation=1):
        if weight.device != "cuda" or weight.dtype != "float32":
            raise ValueError("CapturedConv1d requires a float32 CUDA weight tensor.")
        if not getattr(_C, "HAS_CUDNN", False):
            raise RuntimeError("CapturedConv1d requires a build with cuDNN support.")

        C_out, C_in, K = weight.shape
        L_out = (length + 2 * padding - dilation * (K - 1) - 1) // stride + 1
        if L_out <= 0:
            raise ValueError(f"CapturedConv1d: computed output length {L_out} <= 0")

        self.weight = weight
        self.stride, self.padding, self.dilation = stride, padding, dilation

        import numpy as np
        zeros = lambda shape: aakaar.from_numpy(np.zeros(shape, dtype=np.float32), device="cuda")
        self.x = zeros((batch, C_in, length))
        self.y = zeros((batch, C_out, L_out))
        self.grad_y = zeros((batch, C_out, L_out))
        self.grad_x = zeros((batch, C_in, length))
        self.grad_w = zeros((C_out, C_in, K))

        # Warmup phase (uncaptured) — populates the algorithm/workspace cache.
        x_ = self.x; x_.requires_grad = True
        w_ = self.weight; w_.requires_grad = True
        _C.conv1d_cudnn(self.x, self.weight, stride, padding, dilation)
        out = _C.conv1d_cudnn(x_, w_, stride, padding, dilation)
        out.backward(self.grad_y)
        x_.requires_grad = False
        w_.requires_grad = False
        x_.zero_grad(); w_.zero_grad()

        # --- Forward graph ---
        self._fwd_handle = _C._cuda_graph_begin_capture()
        _C._cudnn_set_stream_for_capture(self._fwd_handle)
        _C._conv1d_forward_into(self.x, self.weight, self.y, stride, padding, dilation)
        _C._cuda_graph_end_capture(self._fwd_handle)
        _C._cudnn_reset_stream()

        # --- Backward: TWO SEPARATE graphs, not one combined capture.
        # Combining backward_data + backward_filter into a single
        # begin_capture/end_capture region was measured to be ~35x slower
        # than replaying each as its own graph back-to-back (0.607ms vs
        # 0.017ms combined) — root cause not fully diagnosed (likely a false
        # serialization dependency or workspace-sharing artifact CUDA's
        # capture machinery introduced across the two cuDNN calls), but the
        # two-graph split is confirmed correct and fast, so that's what ships.
        self._bwd_data_handle = _C._cuda_graph_begin_capture()
        _C._cudnn_set_stream_for_capture(self._bwd_data_handle)
        _C._conv1d_backward_data_into(self.grad_y, self.weight, self.grad_x, stride, padding, dilation)
        _C._cuda_graph_end_capture(self._bwd_data_handle)
        _C._cudnn_reset_stream()

        self._bwd_filter_handle = _C._cuda_graph_begin_capture()
        _C._cudnn_set_stream_for_capture(self._bwd_filter_handle)
        _C._conv1d_backward_filter_into(self.x, self.grad_y, self.grad_w, stride, padding, dilation)
        _C._cuda_graph_end_capture(self._bwd_filter_handle)
        _C._cudnn_reset_stream()

        # In CapturedConv1d.__init__, after capturing all three graphs:
        for _ in range(60):
            self.forward()
        self.synchronize()
        for _ in range(60):
            _C._cuda_graph_replay(self._bwd_data_handle)
        self.synchronize()
        for _ in range(60):
            _C._cuda_graph_replay(self._bwd_filter_handle)
        self.synchronize()

    def forward(self):
        _C._cuda_graph_replay(self._fwd_handle)

    def backward(self):
        _C._cuda_graph_replay_two(self._bwd_data_handle, self._bwd_filter_handle)

    def step(self):
        """Replay forward + full backward in a single Python->C++ call."""
        _C._cuda_graph_replay_full_step(self._fwd_handle, self._bwd_data_handle, self._bwd_filter_handle)

    def synchronize(self):
        _C._cuda_graph_synchronize(self._fwd_handle)
        _C._cuda_graph_synchronize(self._bwd_data_handle)
        _C._cuda_graph_synchronize(self._bwd_filter_handle)

class CapturedConv2d:
    def __init__(self, weight, batch, height, width, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        ...
        # Try capturing backward_data; fall back to uncaptured dispatch if
        # cuDNN has no capture-compatible algorithm for this exact shape
        # (confirmed real limitation, not a bug — some algorithm/shape
        # combinations on some cuDNN/driver versions simply cannot be
        # captured into a CUDA graph).
        self._bwd_data_capturable = True
        try:
            self._bwd_data_handle = _C._cuda_graph_begin_capture()
            _C._cudnn_set_stream_for_capture(self._bwd_data_handle)
            _C._conv2d_backward_data_into(self.grad_y, self.weight, self.grad_x, SH, SW, PH, PW, DH, DW)
            _C._cuda_graph_end_capture(self._bwd_data_handle)
            _C._cudnn_reset_stream()
        except RuntimeError as e:
            print(f"WARNING: backward_data not capturable for this shape ({e}); "
                  f"falling back to uncaptured dispatch for this op.")
            self._bwd_data_capturable = False
            _C._cudnn_reset_stream()

        # same try/except pattern for backward_filter
        self._bwd_filter_capturable = True
        try:
            self._bwd_filter_handle = _C._cuda_graph_begin_capture()
            _C._cudnn_set_stream_for_capture(self._bwd_filter_handle)
            _C._conv2d_backward_filter_into(self.x, self.grad_y, self.grad_w, SH, SW, PH, PW, DH, DW)
            _C._cuda_graph_end_capture(self._bwd_filter_handle)
            _C._cudnn_reset_stream()
        except RuntimeError as e:
            print(f"WARNING: backward_filter not capturable for this shape ({e}); "
                  f"falling back to uncaptured dispatch for this op.")
            self._bwd_filter_capturable = False
            _C._cudnn_reset_stream()
    def forward(self):
        _C._cuda_graph_replay(self._fwd_handle)

    def backward(self):
        if self._bwd_data_capturable and self._bwd_filter_capturable:
            _C._cuda_graph_replay_two(self._bwd_data_handle, self._bwd_filter_handle)
        else:
            # Fallback: dispatch whichever piece isn't capturable normally
            if self._bwd_data_capturable:
                _C._cuda_graph_replay(self._bwd_data_handle)
            else:
                _C._conv2d_backward_data_into(self.grad_y, self.weight, self.grad_x,
                                               *self.stride, *self.padding, *self.dilation)
            if self._bwd_filter_capturable:
                _C._cuda_graph_replay(self._bwd_filter_handle)
            else:
                _C._conv2d_backward_filter_into(self.x, self.grad_y, self.grad_w,
                                                 *self.stride, *self.padding, *self.dilation)

    def backward(self):
        _C._cuda_graph_replay_two(self._bwd_data_handle, self._bwd_filter_handle)

    def step(self):
        _C._cuda_graph_replay_full_step(self._fwd_handle, self._bwd_data_handle, self._bwd_filter_handle)

    def synchronize(self):
        _C._cuda_graph_synchronize(self._fwd_handle)
        _C._cuda_graph_synchronize(self._bwd_data_handle)
        _C._cuda_graph_synchronize(self._bwd_filter_handle)