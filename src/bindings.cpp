#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "tensor.h"
#include <utility>

namespace py = pybind11;

static bool g_grad_enabled = true;
void set_tf32_enabled(bool enabled);
bool get_tf32_enabled();
void set_cudnn_tf32_enabled(bool enabled);
bool get_cudnn_tf32_enabled();
// ---- Forward declarations: raw ops ----
void fill_cpu_random(std::shared_ptr<Tensor> t, unsigned long long seed);
void fill_cpu_randint(std::shared_ptr<Tensor> t, long long low, long long high, unsigned long long seed);
std::shared_ptr<Tensor> run_cpu_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_add_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_sub_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_mul_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_div_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_add_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cpu_sub_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cpu_mul_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cpu_div_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cpu_sum_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cpu_sum_all(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_sum_axis_typed(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cpu_sum_all_typed(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_broadcast_axis(std::shared_ptr<Tensor> a, int dim, int target_size);
std::shared_ptr<Tensor> run_cpu_relu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cpu_sigmoid(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_sigmoid_backward(std::shared_ptr<Tensor> grad_out, const float* sig_out_ptr, int size, std::vector<int> shape);
std::shared_ptr<Tensor> run_cpu_tanh(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_tanh_backward(std::shared_ptr<Tensor> grad_out, const float* tanh_out_ptr, int size, std::vector<int> shape);
std::shared_ptr<Tensor> run_cpu_leaky_relu(std::shared_ptr<Tensor> a, float slope);
std::shared_ptr<Tensor> run_cpu_leaky_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, float slope);
std::shared_ptr<Tensor> run_cpu_exp(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_exp_backward(std::shared_ptr<Tensor> grad_out, const float* exp_out_ptr, int size, std::vector<int> shape);
std::shared_ptr<Tensor> run_cpu_log(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_log_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::pair<std::shared_ptr<Tensor>, std::vector<int>> run_cpu_max_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cpu_max_axis_backward(std::shared_ptr<Tensor> grad_out, const std::vector<int>& argmax,
                                                   std::vector<int> orig_shape, int dim, int reduce_size, int inner_size);
std::string get_openblas_diagnostic();
std::pair<std::shared_ptr<Tensor>, std::vector<int>> run_cpu_max_axis_typed(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cpu_max_axis_backward_typed(std::shared_ptr<Tensor> grad_out, const std::vector<int>& argmax,
                                                         std::vector<int> orig_shape, int dim, int reduce_size, int inner_size);
std::shared_ptr<Tensor> run_cpu_relu_typed(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_relu_backward_typed(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cpu_add_scalar_typed(std::shared_ptr<Tensor> a, double s);
std::shared_ptr<Tensor> run_cpu_sub_scalar_typed(std::shared_ptr<Tensor> a, double s);
std::shared_ptr<Tensor> run_cpu_mul_scalar_typed(std::shared_ptr<Tensor> a, double s);
std::shared_ptr<Tensor> run_cpu_div_scalar_typed(std::shared_ptr<Tensor> a, double s);
std::shared_ptr<Tensor> run_cpu_leaky_relu_typed(std::shared_ptr<Tensor> a, double slope);
std::shared_ptr<Tensor> run_cpu_leaky_relu_backward_typed(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, double slope);
std::shared_ptr<Tensor> run_cpu_matmul_f64(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_matmul_int_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cpu_sqrt(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_sqrt_backward(std::shared_ptr<Tensor> grad_out, const float* sqrt_out_ptr, int size, std::vector<int> shape);
std::shared_ptr<Tensor> run_cpu_sqrt_f64(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_abs(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_abs_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cpu_abs_typed(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_abs_backward_typed(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cpu_sign(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_sign_typed(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cpu_cross_entropy_per_row(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> onehot);
std::shared_ptr<Tensor> run_cpu_cross_entropy_backward(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> onehot, float grad_scale);
std::pair<float, int> run_cpu_bce_forward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target);
std::shared_ptr<Tensor> run_cpu_bce_backward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float grad_scale);
std::pair<float, int> run_cpu_smoothl1_forward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float beta);
std::shared_ptr<Tensor> run_cpu_smoothl1_backward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float beta, float grad_scale);
std::shared_ptr<Tensor> run_cpu_multimargin_per_row(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> onehot, float margin);
std::shared_ptr<Tensor> run_cpu_multimargin_backward(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> onehot, float margin, float grad_scale);
float run_cpu_huber_loss_forward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float delta);
std::shared_ptr<Tensor> run_cpu_huber_loss_backward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float delta, float grad_scale);
std::pair<float, int> run_cpu_mse_forward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target);
std::shared_ptr<Tensor> run_cpu_mse_backward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float grad_scale);
std::shared_ptr<Tensor> run_cpu_nll_per_row(std::shared_ptr<Tensor> log_probs, std::shared_ptr<Tensor> onehot);
std::shared_ptr<Tensor> run_cpu_nll_backward(std::shared_ptr<Tensor> onehot, float grad_scale);
std::pair<float, int> run_cpu_bce_logits_forward(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> target);
std::shared_ptr<Tensor> run_cpu_bce_logits_backward(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> target, float grad_scale);
std::shared_ptr<Tensor> run_cpu_matmul_a_bt(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>);
std::shared_ptr<Tensor> run_cpu_matmul_at_b(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>);
void run_cpu_im2col_2d(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                        int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW, int OH, int OW);
void run_cpu_col2im_2d(std::shared_ptr<Tensor> grad_col, std::shared_ptr<Tensor> grad_x,
                        int B, int C, int H, int W, int KH, int KW, int SH, int SW,
                        int PH, int PW, int DH, int DW, int OH, int OW);
void run_cpu_im2col_2d_typed(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                              int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW, int OH, int OW);
void run_cpu_adam_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                        std::shared_ptr<Tensor> m, std::shared_ptr<Tensor> v,
                        float lr, float beta1, float beta2, float eps,
                        float weight_decay, int t);
void run_cpu_sgd_step(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                       float, float, float, float, int, int, int);
void run_cpu_adamw_step(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                         float, float, float, float, float, int);
void run_cpu_adamax_step(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                          float, float, float, float, float, int);
void run_cpu_nadam_step(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                         float, float, float, float, float, float, float, float, float, int);
void run_cpu_radam_step(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                         float, float, float, float, float, int, float, int);
void run_cpu_rmsprop_step(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                           float, float, float, float, float, int, int);
void run_cpu_adadelta_step(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                            float, float, float, float);
void run_cpu_rprop_step(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                         float, float, float, float, float, int);
void run_cpu_add_channel_bias(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> bias,
                               std::shared_ptr<Tensor> out, int C, int HW, int total);
void run_cpu_channel_bias_grad(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, int, int);  
void run_cpu_im2col_3d(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                        int KD, int KH, int KW, int SD, int SH, int SW,
                        int PD, int PH, int PW, int DD, int DH, int DW,
                        int OD, int OH, int OW);
void run_cpu_col2im_3d(std::shared_ptr<Tensor> grad_col, std::shared_ptr<Tensor> grad_x,
                        int B, int C, int D, int H, int W,
                        int KD, int KH, int KW, int SD, int SH, int SW,
                        int PD, int PH, int PW, int DD, int DH, int DW,
                        int OD, int OH, int OW);                             
void run_cpu_channel_bias_grad_nd(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, int, int, int);

#ifndef AAKAAR_NO_CUDA
#include "allocator.h"
#include "cuda_graph.h"
#include "cudnn_manager.h"
#include "pinned_allocator.h"
void run_curand_uniform(std::shared_ptr<Tensor> t, unsigned long long seed);
void run_curand_randint(std::shared_ptr<Tensor> t, long long low, long long high, unsigned long long seed);
void cuda_synchronize();
std::shared_ptr<Tensor> run_cublas_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_add_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_sub_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_mul_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_div_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_add_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cuda_sub_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cuda_mul_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cuda_div_scalar(std::shared_ptr<Tensor> a, float s);
std::shared_ptr<Tensor> run_cuda_sum_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cuda_sum_all(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_sum_axis_typed(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cuda_sum_all_typed(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_broadcast_axis(std::shared_ptr<Tensor> a, int dim, int target_size);
std::shared_ptr<Tensor> run_cuda_relu(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cuda_sigmoid(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_sigmoid_backward(std::shared_ptr<Tensor> grad_out, const float* sig_out_ptr, int size, std::vector<int> shape);
std::shared_ptr<Tensor> run_cuda_tanh(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_tanh_backward(std::shared_ptr<Tensor> grad_out, const float* tanh_out_ptr, int size, std::vector<int> shape);
std::shared_ptr<Tensor> run_cuda_leaky_relu(std::shared_ptr<Tensor> a, float slope);
std::shared_ptr<Tensor> run_cuda_leaky_relu_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, float slope);
std::shared_ptr<Tensor> run_cuda_exp(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_exp_backward(std::shared_ptr<Tensor> grad_out, const float* exp_out_ptr, int size, std::vector<int> shape);
std::shared_ptr<Tensor> run_cuda_log(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_log_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::pair<std::shared_ptr<Tensor>, std::vector<int>> run_cuda_max_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::pair<std::shared_ptr<Tensor>, std::vector<int>> run_cuda_max_axis_typed(std::shared_ptr<Tensor> a, int dim, bool keepdim);
std::shared_ptr<Tensor> run_cuda_max_axis_backward_typed(std::shared_ptr<Tensor> grad_out, const std::vector<int>& argmax,
                                                          std::vector<int> orig_shape, int dim, int reduce_size, int inner_size);
std::shared_ptr<Tensor> run_cuda_max_axis_backward(std::shared_ptr<Tensor> grad_out, const std::vector<int>& argmax,
                                                    std::vector<int> orig_shape, int dim, int reduce_size, int inner_size);
std::shared_ptr<Tensor> run_cuda_relu_typed(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_relu_backward_typed(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cuda_add_scalar_typed(std::shared_ptr<Tensor> a, double s);
std::shared_ptr<Tensor> run_cuda_sub_scalar_typed(std::shared_ptr<Tensor> a, double s);
std::shared_ptr<Tensor> run_cuda_mul_scalar_typed(std::shared_ptr<Tensor> a, double s);
std::shared_ptr<Tensor> run_cuda_div_scalar_typed(std::shared_ptr<Tensor> a, double s);
std::shared_ptr<Tensor> run_cuda_leaky_relu_typed(std::shared_ptr<Tensor> a, double slope);
std::shared_ptr<Tensor> run_cuda_leaky_relu_backward_typed(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input, double slope);
std::shared_ptr<Tensor> run_cuda_matmul_f64(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_matmul_int_typed(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> run_cuda_sqrt(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_sqrt_backward(std::shared_ptr<Tensor> grad_out, const float* sqrt_out_ptr, int size, std::vector<int> shape);
std::shared_ptr<Tensor> run_cuda_sqrt_f64(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_abs(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_abs_backward(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cuda_abs_typed(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_abs_backward_typed(std::shared_ptr<Tensor> grad_out, std::shared_ptr<Tensor> input);
std::shared_ptr<Tensor> run_cuda_sign(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> run_cuda_sign_typed(std::shared_ptr<Tensor> a);
#if !defined(AAKAAR_NO_CUDA) && defined(AAKAAR_HAS_CUDNN)
std::shared_ptr<Tensor> run_cudnn_conv1d_forward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w,
                                                   int stride, int padding, int dilation, int L_out);
std::shared_ptr<Tensor> run_cudnn_conv1d_backward_data(std::shared_ptr<Tensor> grad_y, std::shared_ptr<Tensor> w,
                                                         int B, int C_in, int L_in, int stride, int padding,
                                                         int dilation, int L_out);
std::shared_ptr<Tensor> run_cudnn_conv1d_backward_filter(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> grad_y,
                                                           int C_out, int K, int stride, int padding,
                                                           int dilation, int L_out);
void run_cudnn_conv1d_forward_into(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w, std::shared_ptr<Tensor> y,
                                    int stride, int padding, int dilation);
void run_cudnn_conv1d_backward_data_into(std::shared_ptr<Tensor> grad_y, std::shared_ptr<Tensor> w,
                                          std::shared_ptr<Tensor> grad_x, int stride, int padding, int dilation);
void run_cudnn_conv1d_backward_filter_into(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> grad_y,
                                            std::shared_ptr<Tensor> grad_w, int stride, int padding, int dilation);
void cudnn_set_stream_for_capture(std::shared_ptr<GraphHandle> handle);
std::shared_ptr<Tensor> run_cudnn_conv2d_forward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w,
                                                   int SH, int SW, int PH, int PW, int DH, int DW,
                                                   int OH, int OW);
std::shared_ptr<Tensor> run_cudnn_conv2d_backward_data(std::shared_ptr<Tensor> grad_y, std::shared_ptr<Tensor> w,
                                                         int B, int C_in, int H, int W, int SH, int SW,
                                                         int PH, int PW, int DH, int DW, int OH, int OW,
                                                         bool require_capturable);
std::shared_ptr<Tensor> run_cudnn_conv2d_backward_filter(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> grad_y,
                                                           int C_out, int KH, int KW, int SH, int SW,
                                                           int PH, int PW, int DH, int DW, int OH, int OW,
                                                           bool require_capturable);
std::shared_ptr<Tensor> run_cudnn_conv3d_forward(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w,
                                                  int SD, int SH, int SW, int PD, int PH, int PW,
                                                  int DD, int DH, int DW, int OD, int OH, int OW);
std::shared_ptr<Tensor> run_cudnn_conv3d_backward_data(std::shared_ptr<Tensor> grad_y, std::shared_ptr<Tensor> w,
                                                        int B, int Cin, int D, int H, int W,
                                                        int SD, int SH, int SW, int PD, int PH, int PW,
                                                        int DD, int DH, int DW, int OD, int OH, int OW);
std::shared_ptr<Tensor> run_cudnn_conv3d_backward_filter(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> grad_y,
                                                          int Cout, int KD, int KH, int KW,
                                                          int SD, int SH, int SW, int PD, int PH, int PW,
                                                          int DD, int DH, int DW, int OD, int OH, int OW);                                                           
void run_cudnn_conv2d_forward_into(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w, std::shared_ptr<Tensor> y,
                                    int SH, int SW, int PH, int PW, int DH, int DW);                                    
void run_cudnn_conv2d_backward_data_into(std::shared_ptr<Tensor> grad_y, std::shared_ptr<Tensor> w,
                                          std::shared_ptr<Tensor> grad_x, int SH, int SW, int PH, int PW, int DH, int DW);
void run_cudnn_conv2d_backward_filter_into(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> grad_y,
                                            std::shared_ptr<Tensor> grad_w, int SH, int SW, int PH, int PW, int DH, int DW);
void cudnn_reset_stream();
#endif
std::pair<float, std::shared_ptr<Tensor>> run_cuda_huber_loss_forward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float delta);
std::shared_ptr<Tensor> run_cuda_huber_loss_backward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float delta, float grad_scale);
std::shared_ptr<Tensor> run_cuda_cross_entropy_per_row(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> onehot);
std::shared_ptr<Tensor> run_cuda_cross_entropy_backward(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> onehot, float grad_scale);
std::pair<float, int> run_cuda_bce_forward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target);
std::shared_ptr<Tensor> run_cuda_bce_backward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float grad_scale);
std::pair<float, int> run_cuda_smoothl1_forward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float beta);
std::shared_ptr<Tensor> run_cuda_smoothl1_backward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float beta, float grad_scale);
std::shared_ptr<Tensor> run_cuda_multimargin_per_row(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> onehot, float margin);
std::shared_ptr<Tensor> run_cuda_multimargin_backward(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> onehot, float margin, float grad_scale);
std::pair<float, int> run_cuda_mse_forward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target);
std::shared_ptr<Tensor> run_cuda_mse_backward(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float grad_scale);
std::shared_ptr<Tensor> run_cuda_nll_per_row(std::shared_ptr<Tensor> log_probs, std::shared_ptr<Tensor> onehot);
std::shared_ptr<Tensor> run_cuda_nll_backward(std::shared_ptr<Tensor> onehot, float grad_scale);
std::pair<float, int> run_cuda_bce_logits_forward(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> target);
std::shared_ptr<Tensor> run_cuda_bce_logits_backward(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> target, float grad_scale);
std::shared_ptr<Tensor> run_cublas_matmul_a_bt(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>);
std::shared_ptr<Tensor> run_cublas_matmul_at_b(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>);
void run_cuda_im2col_2d(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                         int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW, int OH, int OW);
void run_cuda_col2im_2d(std::shared_ptr<Tensor> grad_col, std::shared_ptr<Tensor> grad_x,
                         int B, int C, int H, int W, int KH, int KW, int SH, int SW,
                         int PH, int PW, int DH, int DW, int OH, int OW);
void run_cuda_im2col_2d_typed(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                               int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW, int OH, int OW);
void run_cuda_adam_step(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                         std::shared_ptr<Tensor> m, std::shared_ptr<Tensor> v,
                         float lr, float beta1, float beta2, float eps,
                         float weight_decay, int t);
void run_cuda_sgd_step(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                        float, float, float, float, int, int, int);
void run_cuda_adamw_step(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                          float, float, float, float, float, int);
void run_cuda_adamax_step(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                           float, float, float, float, float, int);
void run_cuda_nadam_step(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                          float, float, float, float, float, float, float, float, float, int);
void run_cuda_radam_step(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                          float, float, float, float, float, int, float, int);
void run_cuda_rmsprop_step(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                            float, float, float, float, float, int, int);
void run_cuda_adadelta_step(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                             float, float, float, float);
void run_cuda_rprop_step(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>,
                          float, float, float, float, float, int);
void run_cuda_add_channel_bias(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> bias,
                                std::shared_ptr<Tensor> out, int C, int HW, int total);
void run_cuda_channel_bias_grad(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, int, int); 
void run_cuda_im2col_3d(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> col,
                         int KD, int KH, int KW, int SD, int SH, int SW,
                         int PD, int PH, int PW, int DD, int DH, int DW,
                         int OD, int OH, int OW);
void run_cuda_col2im_3d(std::shared_ptr<Tensor> grad_col, std::shared_ptr<Tensor> grad_x,
                         int B, int C, int D, int H, int W,
                         int KD, int KH, int KW, int SD, int SH, int SW,
                         int PD, int PH, int PW, int DD, int DH, int DW,
                         int OD, int OH, int OW);                      
void run_cuda_channel_bias_grad_nd(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, int, int, int);                                                                  
void empty_cache() { CachingAllocator::get_instance().empty_cache(); }
#endif

// ---- Forward declarations: autograd-aware dispatch ----
static std::shared_ptr<Tensor> dispatch_add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
static std::shared_ptr<Tensor> dispatch_sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
static std::shared_ptr<Tensor> dispatch_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
static std::shared_ptr<Tensor> dispatch_div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
static std::shared_ptr<Tensor> dispatch_add_scalar(std::shared_ptr<Tensor> a, float s);
static std::shared_ptr<Tensor> dispatch_sub_scalar(std::shared_ptr<Tensor> a, float s);
static std::shared_ptr<Tensor> dispatch_mul_scalar(std::shared_ptr<Tensor> a, float s);
static std::shared_ptr<Tensor> dispatch_div_scalar(std::shared_ptr<Tensor> a, float s);
static std::shared_ptr<Tensor> dispatch_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
static std::shared_ptr<Tensor> dispatch_sum_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim);
static std::shared_ptr<Tensor> dispatch_broadcast_axis(std::shared_ptr<Tensor> a, int dim, int target_size);
static std::shared_ptr<Tensor> dispatch_contiguous(std::shared_ptr<Tensor> a);
static std::shared_ptr<Tensor> dispatch_huber_loss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float delta);
static std::shared_ptr<Tensor> dispatch_im2col_2d(std::shared_ptr<Tensor> x, int KH, int KW, int SH, int SW, int PH, int PW, int DH, int DW, int OH, int OW);
static std::shared_ptr<Tensor> dispatch_im2col_3d(std::shared_ptr<Tensor> x, int KD, int KH, int KW, int SD, int SH, int SW, int PD, int PH, int PW, int DD, int DH, int DW, int OD, int OH, int OW);

static bool is_channel_bias_shape(const std::vector<int>& b_shape, const std::vector<int>& a_shape, int& C, int& HW) {
    // Matches (1, C, 1, 1) against (B, C, H, W), or (1, C, 1) against (B, C, L).
    if (b_shape.size() != a_shape.size()) return false;
    if (b_shape.size() < 3) return false;
    if (b_shape[0] != 1 || b_shape[1] != a_shape[1]) return false;
    for (size_t i = 2; i < b_shape.size(); ++i) if (b_shape[i] != 1) return false;
    C = a_shape[1];
    HW = 1;
    for (size_t i = 2; i < a_shape.size(); ++i) HW *= a_shape[i];
    return true;
}

// Add near your other includes/declarations
int cuda_device_count() {
#ifdef AAKAAR_NO_CUDA
    return 0;
#else
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return 0;  // no driver, no GPU, etc. — not an error condition, just "unavailable"
    return count;
#endif
}

bool cuda_is_available() {
    return cuda_device_count() > 0;
}

// ---- Dispatch implementations ----
static std::shared_ptr<Tensor> dispatch_leaky_relu(std::shared_ptr<Tensor> a, double slope) {
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                    "'. Only float32 currently supports gradients.");
        if (!a->is_contiguous()) a = dispatch_contiguous(a);
        std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") result = run_cuda_leaky_relu_typed(a, slope);
        else
#endif
        result = run_cpu_leaky_relu_typed(a, slope);
        return result;
    }
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_leaky_relu(a, (float)slope);
    else
#endif
    result = run_cpu_leaky_relu(a, (float)slope);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "leaky_relu";
        auto slope_f = (float)slope;
        node->backward_fn = [a, slope_f](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (a->device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_leaky_relu_backward(grad_out, a, slope_f)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_leaky_relu_backward(grad_out, a, slope_f)};
        };
        result->grad_fn = node;
    }
    return result;
}
static std::shared_ptr<Tensor> dispatch_max_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim) {
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
        if (!a->is_contiguous()) a = dispatch_contiguous(a);

        int ndim = (int)a->shape.size();
        int norm_dim = dim < 0 ? dim + ndim : dim;

#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") { auto pr = run_cuda_max_axis_typed(a, dim, keepdim); return pr.first; }
#endif
        auto pr = run_cpu_max_axis_typed(a, dim, keepdim);
        return pr.first;
    }
    if (!a->is_contiguous()) a = dispatch_contiguous(a);    
    int ndim = (int)a->shape.size();
    int norm_dim = dim < 0 ? dim + ndim : dim;
    int reduce_size = a->shape[norm_dim];
    int inner_size = 1;
    for (int i = norm_dim + 1; i < ndim; ++i) inner_size *= a->shape[i];

    std::shared_ptr<Tensor> result;
    std::vector<int> argmax;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") { auto pr = run_cuda_max_axis(a, dim, keepdim); result = pr.first; argmax = pr.second; }
    else
#endif
    { auto pr = run_cpu_max_axis(a, dim, keepdim); result = pr.first; argmax = pr.second; }

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "max_axis";
        auto orig_shape = a->shape;
        auto dev = a->device;
        node->backward_fn = [argmax, orig_shape, norm_dim, reduce_size, inner_size, dev]
                             (std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (dev == "cuda") return std::vector<std::shared_ptr<Tensor>>{
                run_cuda_max_axis_backward(grad_out, argmax, orig_shape, norm_dim, reduce_size, inner_size)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{
                run_cpu_max_axis_backward(grad_out, argmax, orig_shape, norm_dim, reduce_size, inner_size)};
        };
        result->grad_fn = node;
    }
    return result;
}
static std::shared_ptr<Tensor> dispatch_exp(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_exp(a);
    else
#endif
    result = run_cpu_exp(a);
    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "exp";
        
        float* out_ptr = result->fptr();
        int out_size = result->size;
        auto out_shape = result->shape;
        auto out_device = result->device;
        
        node->backward_fn = [out_ptr, out_size, out_shape, out_device](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (out_device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_exp_backward(grad_out, out_ptr, out_size, out_shape)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_exp_backward(grad_out, out_ptr, out_size, out_shape)};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_log(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_log(a);
    else
#endif
    result = run_cpu_log(a);
    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "log";
        node->backward_fn = [a](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (a->device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_log_backward(grad_out, a)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_log_backward(grad_out, a)};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_add_scalar(std::shared_ptr<Tensor> a, float s) {
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_add_scalar_typed(a, (double)s);
#endif
        return run_cpu_add_scalar_typed(a, (double)s);
    }

    // --- THE MISSING COMPUTATION ---
    // We must declare and compute 'result' for FLOAT32 before doing autograd wiring
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") {
        // Use your specific float32 CUDA function here (or the typed one if they share it)
        result = run_cuda_add_scalar(a, s); 
    } else 
#endif
    {
        result = run_cpu_add_scalar(a, s);
    }
    // -------------------------------

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "add_scalar";
        node->backward_fn = [](std::shared_ptr<Tensor> grad_out) {
            return std::vector<std::shared_ptr<Tensor>>{grad_out};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_sub_scalar(std::shared_ptr<Tensor> a, float s) {
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_sub_scalar_typed(a, (double)s);
#endif
        return run_cpu_sub_scalar_typed(a, (double)s);
    }

    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_sub_scalar(a, s);
    else
#endif
    result = run_cpu_sub_scalar(a, s);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "sub_scalar";
        node->backward_fn = [](std::shared_ptr<Tensor> grad_out) {
            return std::vector<std::shared_ptr<Tensor>>{grad_out};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_mul_scalar(std::shared_ptr<Tensor> a, float s) {
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_mul_scalar_typed(a, (double)s);
#endif
        return run_cpu_mul_scalar_typed(a, (double)s);
    }

    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_mul_scalar(a, s);
    else
#endif
    result = run_cpu_mul_scalar(a, s);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "mul_scalar";
        node->backward_fn = [s](std::shared_ptr<Tensor> grad_out) {
            auto da = dispatch_mul_scalar(grad_out, s);
            return std::vector<std::shared_ptr<Tensor>>{da};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_div_scalar(std::shared_ptr<Tensor> a, float s) {
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_div_scalar_typed(a, (double)s);
#endif
        return run_cpu_div_scalar_typed(a, (double)s);
    }

    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_div_scalar(a, s);
    else
#endif
    result = run_cpu_div_scalar(a, s);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "div_scalar";
        node->backward_fn = [s](std::shared_ptr<Tensor> grad_out) {
            auto da = dispatch_div_scalar(grad_out, s);
            return std::vector<std::shared_ptr<Tensor>>{da};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_relu(std::shared_ptr<Tensor> a) {
    if (a->dtype != DType::FLOAT32) {
        if (!a->is_contiguous()) a = dispatch_contiguous(a);
        std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") result = run_cuda_relu_typed(a);
        else
#endif
        result = run_cpu_relu_typed(a);

        if (g_grad_enabled && a->requires_grad) {
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
        }
        return result;
    }
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_relu(a);
    else
#endif
    result = run_cpu_relu(a);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "relu";
        node->backward_fn = [a](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (a->device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_relu_backward(grad_out, a)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_relu_backward(grad_out, a)};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_sqrt(std::shared_ptr<Tensor> a) {
    if (a->dtype == DType::FLOAT64) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype 'float64'. Only float32 currently supports gradients.");
        if (!a->is_contiguous()) a = dispatch_contiguous(a);
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_sqrt_f64(a);
#endif
        return run_cpu_sqrt_f64(a);
    }
    if (a->dtype == DType::INT32 || a->dtype == DType::INT64) {
        throw std::runtime_error("sqrt(): dtype '" + dtype_name(a->dtype) + "' is not supported. Use a float dtype.");
    }
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_sqrt(a);
    else
#endif
    result = run_cpu_sqrt(a);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "sqrt";
        float* out_ptr = result->fptr();
        int out_size = result->size;
        auto out_shape = result->shape;
        auto out_device = result->device;
        node->backward_fn = [out_ptr, out_size, out_shape, out_device](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (out_device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_sqrt_backward(grad_out, out_ptr, out_size, out_shape)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_sqrt_backward(grad_out, out_ptr, out_size, out_shape)};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_abs(std::shared_ptr<Tensor> a) {
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) + "'. Only float32 currently supports gradients.");
        if (!a->is_contiguous()) a = dispatch_contiguous(a);
        std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") result = run_cuda_abs_typed(a);
        else
#endif
        result = run_cpu_abs_typed(a);
        return result;
    }
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_abs(a);
    else
#endif
    result = run_cpu_abs(a);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "abs";
        node->backward_fn = [a](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (a->device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_abs_backward(grad_out, a)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_abs_backward(grad_out, a)};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_sigmoid(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_sigmoid(a);
    else
#endif
    result = run_cpu_sigmoid(a);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "sigmoid";
        // Capture raw pointer/shape only — NOT `result` itself, which would
        // create a shared_ptr cycle (result->grad_fn->backward_fn->result)
        // that leaks memory forever since refcounting can't collect cycles.
        float* out_ptr = result->fptr();
        int out_size = result->size;
        auto out_shape = result->shape;
        auto out_device = result->device;
        node->backward_fn = [out_ptr, out_size, out_shape, out_device](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (out_device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_sigmoid_backward(grad_out, out_ptr, out_size, out_shape)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_sigmoid_backward(grad_out, out_ptr, out_size, out_shape)};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_tanh(std::shared_ptr<Tensor> a) {
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_tanh(a);
    else
#endif
    result = run_cpu_tanh(a);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "tanh";
        
        float* out_ptr = result->fptr();
        int out_size = result->size;
        auto out_shape = result->shape;
        auto out_device = result->device;
        
        node->backward_fn = [out_ptr, out_size, out_shape, out_device](std::shared_ptr<Tensor> grad_out) {
#ifndef AAKAAR_NO_CUDA
            if (out_device == "cuda") return std::vector<std::shared_ptr<Tensor>>{run_cuda_tanh_backward(grad_out, out_ptr, out_size, out_shape)};
#endif
            return std::vector<std::shared_ptr<Tensor>>{run_cpu_tanh_backward(grad_out, out_ptr, out_size, out_shape)};
        };
        result->grad_fn = node;
    }
    return result;
}

// ---- backward() driver: reverse-mode topological traversal ----

static void tensor_backward(std::shared_ptr<Tensor> root, std::shared_ptr<Tensor> grad_output, bool retain_graph) {
    if (!grad_output) {
        if (root->size != 1)
            throw std::runtime_error("backward() requires an explicit gradient for non-scalar tensors");
        grad_output = std::make_shared<Tensor>(root->shape, root->device);
        float one = 1.0f;
#ifndef AAKAAR_NO_CUDA
        if (root->device == "cuda") {
            cudaMemcpy(grad_output->fptr(), &one, sizeof(float), cudaMemcpyHostToDevice);
        } else
#endif
        {
            grad_output->fptr()[0] = one;
        }
    }

    std::vector<std::shared_ptr<Tensor>> topo;
    std::unordered_set<Tensor*> visited;
    std::function<void(std::shared_ptr<Tensor>)> dfs = [&](std::shared_ptr<Tensor> t) {
        if (visited.count(t.get())) return;
        visited.insert(t.get());
        if (t->grad_fn) {
            for (auto& inp : t->grad_fn->inputs) dfs(inp);
        }
        topo.push_back(t);
    };
    dfs(root);

    std::unordered_map<Tensor*, std::shared_ptr<Tensor>> grad_map;
    grad_map[root.get()] = grad_output;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        auto t = *it;
        auto grad_it = grad_map.find(t.get());
        if (grad_it == grad_map.end()) continue;
        auto g = grad_it->second;

        if (t->requires_grad && !t->grad_fn) {
            if (!t->grad) t->grad = g;
            else t->grad = dispatch_add(t->grad, g);
        }

        if (t->grad_fn) {
            if (t->grad_fn->freed) {
                throw std::runtime_error(
                    "Trying to backward through the graph a second time (or a part of it), but the "
                    "intermediate results needed have already been freed. Pass retain_graph=True to "
                    "backward() the first time if you need to backward through this part of the graph "
                    "more than once."
                );
            }
            auto input_grads = t->grad_fn->backward_fn(g);
            for (size_t i = 0; i < t->grad_fn->inputs.size(); ++i) {
                auto& inp = t->grad_fn->inputs[i];
                if (!inp->requires_grad) continue;
                auto existing = grad_map.find(inp.get());
                if (existing == grad_map.end()) grad_map[inp.get()] = input_grads[i];
                else grad_map[inp.get()] = dispatch_add(existing->second, input_grads[i]);
            }
        }
    }

    if (!retain_graph) {
        for (auto& t : topo) {
            if (t->grad_fn) t->grad_fn->freed = true;
        }
    }
}

static std::shared_ptr<Tensor> dispatch_broadcast_axis(std::shared_ptr<Tensor> a, int dim, int target_size) {
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") return run_cuda_broadcast_axis(a, dim, target_size);
#endif
    return run_cpu_broadcast_axis(a, dim, target_size);
}
static std::shared_ptr<Tensor> dispatch_contiguous(std::shared_ptr<Tensor> a) {
    auto result = a->contiguous();
    if (g_grad_enabled && a->requires_grad && result.get() != a.get()) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "contiguous";
        node->backward_fn = [](std::shared_ptr<Tensor> grad_out) {
            return std::vector<std::shared_ptr<Tensor>>{grad_out};
        };
        result->grad_fn = node;
    }
    return result;
}
static std::shared_ptr<Tensor> dispatch_sum_axis(std::shared_ptr<Tensor> a, int dim, bool keepdim) {
        if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_sum_axis_typed(a, dim, keepdim);
#endif
        return run_cpu_sum_axis_typed(a, dim, keepdim);
    }
    if (!a->is_contiguous()) {
        a = a->contiguous();  // auto-materialize, matching torch's ergonomic sum() behavior
    }
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_sum_axis(a, dim, keepdim);
    else
#endif
    result = run_cpu_sum_axis(a, dim, keepdim);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "sum_axis";
        int norm_dim = dim < 0 ? dim + (int)a->shape.size() : dim;
        int original_size = a->shape[norm_dim];
        node->backward_fn = [norm_dim, original_size, keepdim](std::shared_ptr<Tensor> grad_out) {
            auto g = grad_out;
            if (!keepdim) {
                // The forward pass dropped this axis entirely (keepdim=False), so
                // grad_out is missing it too. Reinsert a size-1 axis at norm_dim
                // before broadcasting, since broadcast_axis expects that axis to
                // already exist (as size 1) in order to expand it back out.
                auto reshaped = g->shape;
                reshaped.insert(reshaped.begin() + norm_dim, 1);
                g = g->reshape(reshaped);
            }
            auto expanded = dispatch_broadcast_axis(g, norm_dim, original_size);
            return std::vector<std::shared_ptr<Tensor>>{expanded};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_sum_all(std::shared_ptr<Tensor> a) {
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_sum_all_typed(a);
#endif
        return run_cpu_sum_all_typed(a);
    }
    if (!a->is_contiguous()) {
        a = dispatch_contiguous(a);  // auto-materialize, matching torch's ergonomic sum() behavior
    }
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_sum_all(a);
    else
#endif
    result = run_cpu_sum_all(a);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "sum_all";
        auto orig_shape = a->shape;
        node->backward_fn = [orig_shape](std::shared_ptr<Tensor> grad_out) {
            // grad_out is shape (1,). Reshape it to all-ones matching orig_shape's rank,
            // then broadcast each axis up to its original size in turn.
            std::vector<int> ones_shape(orig_shape.size(), 1);
            auto g = grad_out->reshape(ones_shape);
            for (size_t d = 0; d < orig_shape.size(); ++d) {
                if (orig_shape[d] > 1) {
                    g = dispatch_broadcast_axis(g, (int)d, orig_shape[d]);
                }
            }
            return std::vector<std::shared_ptr<Tensor>>{g};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_im2col_1d(std::shared_ptr<Tensor> x, int kernel_size, int stride,
                                                    int padding, int dilation, int out_length) {
    if (x->shape.size() != 3)
        throw std::invalid_argument("im2col_1d: expected input of shape (batch, channels, length), got rank " +
                                     std::to_string(x->shape.size()));
    if (!x->is_contiguous()) x = dispatch_contiguous(x);

    int B = x->shape[0], C = x->shape[1], L_in = x->shape[2];
    std::vector<int> col_shape = {B, C * kernel_size, out_length};

    if (x->dtype != DType::FLOAT32) {
        if (g_grad_enabled && x->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(x->dtype) +
                                      "'. Only float32 currently supports gradients.");
        auto result = std::make_shared<Tensor>(col_shape, x->device, x->dtype);
#ifndef AAKAAR_NO_CUDA
        if (x->device == "cuda") {
            run_cuda_im2col_1d_typed(x, result, kernel_size, stride, padding, dilation, out_length);
            return result;
        }
#endif
        run_cpu_im2col_1d_typed(x, result, kernel_size, stride, padding, dilation, out_length);
        return result;
    }

    auto result = std::make_shared<Tensor>(col_shape, x->device);
#ifndef AAKAAR_NO_CUDA
    if (x->device == "cuda") run_cuda_im2col_1d(x, result, kernel_size, stride, padding, dilation, out_length);
    else
#endif
    run_cpu_im2col_1d(x, result, kernel_size, stride, padding, dilation, out_length);

    if (g_grad_enabled && x->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {x};
        node->op_name = "im2col_1d";
        int B_c = B, C_c = C, L_in_c = L_in, K_c = kernel_size, S_c = stride, P_c = padding, D_c = dilation, Lo_c = out_length;
        auto dev = x->device;
        node->backward_fn = [B_c, C_c, L_in_c, K_c, S_c, P_c, D_c, Lo_c, dev](std::shared_ptr<Tensor> grad_out) {
            auto grad_x = std::make_shared<Tensor>(std::vector<int>{B_c, C_c, L_in_c}, dev);
#ifndef AAKAAR_NO_CUDA
            if (dev == "cuda") run_cuda_col2im_1d(grad_out, grad_x, B_c, C_c, L_in_c, K_c, S_c, P_c, D_c, Lo_c);
            else
#endif
            run_cpu_col2im_1d(grad_out, grad_x, B_c, C_c, L_in_c, K_c, S_c, P_c, D_c, Lo_c);
            return std::vector<std::shared_ptr<Tensor>>{grad_x};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_col2im_1d(std::shared_ptr<Tensor> col, int C, int L_in, int stride,
                                                     int padding, int dilation, int L_out) {
    if (col->shape.size() != 3)
        throw std::invalid_argument("col2im_1d: expected input of shape (batch, channels*kernel, out_length), got rank " +
                                     std::to_string(col->shape.size()));
    if (col->dtype != DType::FLOAT32)
        throw std::runtime_error("col2im_1d: only float32 is currently supported.");
    if (!col->is_contiguous()) col = dispatch_contiguous(col);

    int B = col->shape[0];
    int CK = col->shape[1];
    if (CK % C != 0)
        throw std::invalid_argument("col2im_1d: channels*kernel_size (" + std::to_string(CK) +
                                     ") is not divisible by C (" + std::to_string(C) + ")");
    int K = CK / C;

    auto result = std::make_shared<Tensor>(std::vector<int>{B, C, L_in}, col->device);
#ifndef AAKAAR_NO_CUDA
    if (col->device == "cuda")
        run_cuda_col2im_1d(col, result, B, C, L_in, K, stride, padding, dilation, L_out);
    else
#endif
    run_cpu_col2im_1d(col, result, B, C, L_in, K, stride, padding, dilation, L_out);

    if (g_grad_enabled && col->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {col};
        node->op_name = "col2im_1d";
        int K_c = K, S_c = stride, P_c = padding, D_c = dilation, Lo_c = L_out;
        node->backward_fn = [K_c, S_c, P_c, D_c, Lo_c](std::shared_ptr<Tensor> grad_out) {
            // col2im's backward is exactly im2col with the same params:
            // gather back out the same (K positions x L_out) windows that
            // col2im scattered from — the precise inverse relationship.
            auto grad_col = dispatch_im2col_1d(grad_out, K_c, S_c, P_c, D_c, Lo_c);
            return std::vector<std::shared_ptr<Tensor>>{grad_col};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_col2im_2d(std::shared_ptr<Tensor> col, int C, int H_in, int W_in,
                                                     int SH, int SW, int PH, int PW, int DH, int DW,
                                                     int OH, int OW) {
    if (col->shape.size() != 3)
        throw std::invalid_argument("col2im_2d: expected input of shape (batch, channels*KH*KW, OH*OW), got rank " +
                                     std::to_string(col->shape.size()));
    if (col->dtype != DType::FLOAT32)
        throw std::runtime_error("col2im_2d: only float32 is currently supported.");
    if (!col->is_contiguous()) col = dispatch_contiguous(col);

    int B = col->shape[0];
    int CKK = col->shape[1];
    if (CKK % C != 0)
        throw std::invalid_argument("col2im_2d: channels*KH*KW is not divisible by C");
    int KK = CKK / C;
    int K = (int)std::round(std::sqrt((double)KK));  // assumes square kernel, matches Conv2d's usage
    if (K * K != KK)
        throw std::invalid_argument("col2im_2d: cannot infer square kernel size from channel*kernel dimension");

    auto result = std::make_shared<Tensor>(std::vector<int>{B, C, H_in, W_in}, col->device);
#ifndef AAKAAR_NO_CUDA
    if (col->device == "cuda")
        run_cuda_col2im_2d(col, result, B, C, H_in, W_in, K, K, SH, SW, PH, PW, DH, DW, OH, OW);
    else
#endif
    run_cpu_col2im_2d(col, result, B, C, H_in, W_in, K, K, SH, SW, PH, PW, DH, DW, OH, OW);

    if (g_grad_enabled && col->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {col};
        node->op_name = "col2im_2d";
        int K_c = K, SH_c = SH, SW_c = SW, PH_c = PH, PW_c = PW, DH_c = DH, DW_c = DW, OH_c = OH, OW_c = OW;
        node->backward_fn = [K_c, SH_c, SW_c, PH_c, PW_c, DH_c, DW_c, OH_c, OW_c](std::shared_ptr<Tensor> grad_out) {
            // Exact inverse relationship, same as col2im_1d: col2im's
            // backward is im2col with the same params.
            auto grad_col = dispatch_im2col_2d(grad_out, K_c, K_c, SH_c, SW_c, PH_c, PW_c, DH_c, DW_c, OH_c, OW_c);
            return std::vector<std::shared_ptr<Tensor>>{grad_col};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_col2im_3d(std::shared_ptr<Tensor> col, int C, int D_in, int H_in, int W_in,
                                                     int SD, int SH, int SW, int PD, int PH, int PW,
                                                     int DD, int DH, int DW, int OD, int OH, int OW) {
    if (col->shape.size() != 3)
        throw std::invalid_argument("col2im_3d: expected input of shape (batch, channels*KD*KH*KW, OD*OH*OW), got rank " +
                                     std::to_string(col->shape.size()));
    if (col->dtype != DType::FLOAT32)
        throw std::runtime_error("col2im_3d: only float32 is currently supported.");
    if (!col->is_contiguous()) col = dispatch_contiguous(col);

    int B = col->shape[0];
    int CKKK = col->shape[1];
    if (CKKK % C != 0)
        throw std::invalid_argument("col2im_3d: channels*KD*KH*KW is not divisible by C");
    int KKK = CKKK / C;
    int K = (int)std::round(std::cbrt((double)KKK));  // assumes cubic kernel, matches Conv3d's usage
    if (K * K * K != KKK)
        throw std::invalid_argument("col2im_3d: cannot infer cubic kernel size from channel*kernel dimension");

    auto result = std::make_shared<Tensor>(std::vector<int>{B, C, D_in, H_in, W_in}, col->device);
#ifndef AAKAAR_NO_CUDA
    if (col->device == "cuda")
        run_cuda_col2im_3d(col, result, B, C, D_in, H_in, W_in, K, K, K, SD, SH, SW, PD, PH, PW, DD, DH, DW, OD, OH, OW);
    else
#endif
    run_cpu_col2im_3d(col, result, B, C, D_in, H_in, W_in, K, K, K, SD, SH, SW, PD, PH, PW, DD, DH, DW, OD, OH, OW);

    if (g_grad_enabled && col->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {col};
        node->op_name = "col2im_3d";
        int K_c=K, SD_c=SD, SH_c=SH, SW_c=SW, PD_c=PD, PH_c=PH, PW_c=PW, DD_c=DD, DH_c=DH, DW_c=DW, OD_c=OD, OH_c=OH, OW_c=OW;
        node->backward_fn = [K_c,SD_c,SH_c,SW_c,PD_c,PH_c,PW_c,DD_c,DH_c,DW_c,OD_c,OH_c,OW_c](std::shared_ptr<Tensor> grad_out) {
            auto grad_col = dispatch_im2col_3d(grad_out, K_c,K_c,K_c, SD_c,SH_c,SW_c, PD_c,PH_c,PW_c, DD_c,DH_c,DW_c, OD_c,OH_c,OW_c);
            return std::vector<std::shared_ptr<Tensor>>{grad_col};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_im2col_2d(std::shared_ptr<Tensor> x, int KH, int KW, int SH, int SW,
                                                    int PH, int PW, int DH, int DW, int OH, int OW) {
    if (x->shape.size() != 4)
        throw std::invalid_argument("im2col_2d: expected input of shape (batch, channels, H, W), got rank " +
                                     std::to_string(x->shape.size()));
    if (!x->is_contiguous()) x = dispatch_contiguous(x);

    int B = x->shape[0], C = x->shape[1];
    std::vector<int> col_shape = {B, C * KH * KW, OH * OW};

    if (x->dtype != DType::FLOAT32) {
        if (g_grad_enabled && x->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(x->dtype) +
                                      "'. Only float32 currently supports gradients.");
        auto result = std::make_shared<Tensor>(col_shape, x->device, x->dtype);
#ifndef AAKAAR_NO_CUDA
        if (x->device == "cuda") {
            run_cuda_im2col_2d_typed(x, result, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW);
            return result;
        }
#endif
        run_cpu_im2col_2d_typed(x, result, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW);
        return result;
    }

    auto result = std::make_shared<Tensor>(col_shape, x->device);
#ifndef AAKAAR_NO_CUDA
    if (x->device == "cuda") run_cuda_im2col_2d(x, result, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW);
    else
#endif
    run_cpu_im2col_2d(x, result, KH, KW, SH, SW, PH, PW, DH, DW, OH, OW);

    if (g_grad_enabled && x->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {x};
        node->op_name = "im2col_2d";
        int B_c = B, C_c = C, H_c = x->shape[2], W_c = x->shape[3];
        int KH_c = KH, KW_c = KW, SH_c = SH, SW_c = SW, PH_c = PH, PW_c = PW, DH_c = DH, DW_c = DW, OH_c = OH, OW_c = OW;
        auto dev = x->device;
        node->backward_fn = [B_c, C_c, H_c, W_c, KH_c, KW_c, SH_c, SW_c, PH_c, PW_c, DH_c, DW_c, OH_c, OW_c, dev]
                             (std::shared_ptr<Tensor> grad_out) {
            auto grad_x = std::make_shared<Tensor>(std::vector<int>{B_c, C_c, H_c, W_c}, dev);
#ifndef AAKAAR_NO_CUDA
            if (dev == "cuda") run_cuda_col2im_2d(grad_out, grad_x, B_c, C_c, H_c, W_c, KH_c, KW_c, SH_c, SW_c, PH_c, PW_c, DH_c, DW_c, OH_c, OW_c);
            else
#endif
            run_cpu_col2im_2d(grad_out, grad_x, B_c, C_c, H_c, W_c, KH_c, KW_c, SH_c, SW_c, PH_c, PW_c, DH_c, DW_c, OH_c, OW_c);
            return std::vector<std::shared_ptr<Tensor>>{grad_x};
        };
        result->grad_fn = node;
    }
    return result;
}

#ifdef AAKAAR_HAS_CUDNN
static std::shared_ptr<Tensor> dispatch_conv1d_cudnn(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w,
                                                       int stride, int padding, int dilation) {
    if (x->device != "cuda" || w->device != "cuda")
        throw std::runtime_error("conv1d_cudnn: both input and weight must be on device='cuda'.");
    if (x->dtype != DType::FLOAT32 || w->dtype != DType::FLOAT32)
        throw std::runtime_error(
            "conv1d_cudnn: only float32 is supported. cuDNN's double-precision conv support is "
            "unreliable/slow across versions and it has no integer conv path at all — aakaar "
            "intentionally does not route those dtypes here. Use conv1d_im2col (matmul-based) instead.");
    if (!x->is_contiguous()) x = dispatch_contiguous(x);
    if (!w->is_contiguous()) w = dispatch_contiguous(w);

    int B = x->shape[0], C_in = x->shape[1], L_in = x->shape[2];
    int C_out = w->shape[0], K = w->shape[2];
    int L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    if (L_out <= 0)
        throw std::invalid_argument("conv1d_cudnn: computed output length <= 0.");

    auto y = run_cudnn_conv1d_forward(x, w, stride, padding, dilation, L_out);

    if (g_grad_enabled && (x->requires_grad || w->requires_grad)) {
        y->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {x, w};
        node->op_name = "conv1d_cudnn";
        int B_c = B, Cin_c = C_in, Lin_c = L_in, Cout_c = C_out, K_c = K, S_c = stride, P_c = padding, D_c = dilation, Lo_c = L_out;
        node->backward_fn = [x, w, B_c, Cin_c, Lin_c, Cout_c, K_c, S_c, P_c, D_c, Lo_c]
                             (std::shared_ptr<Tensor> grad_out) {
            auto grad_x = run_cudnn_conv1d_backward_data(grad_out, w, B_c, Cin_c, Lin_c, S_c, P_c, D_c, Lo_c);
            auto grad_w = run_cudnn_conv1d_backward_filter(x, grad_out, Cout_c, K_c, S_c, P_c, D_c, Lo_c);
            return std::vector<std::shared_ptr<Tensor>>{grad_x, grad_w};
        };
        y->grad_fn = node;
    }
    return y;
}

static std::shared_ptr<Tensor> dispatch_conv1d_transpose_cudnn(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w,
                                                                 int stride, int padding, int dilation, int L_out) {
    if (x->device != "cuda" || w->device != "cuda")
        throw std::runtime_error("conv1d_transpose_cudnn: both input and weight must be on device='cuda'.");
    if (x->dtype != DType::FLOAT32 || w->dtype != DType::FLOAT32)
        throw std::runtime_error("conv1d_transpose_cudnn: only float32 is supported.");
    if (!x->is_contiguous()) x = dispatch_contiguous(x);
    if (!w->is_contiguous()) w = dispatch_contiguous(w);

    // weight shape: (Cin_t, Cout_t, K) — ConvTranspose1d's declared layout
    // (note: REVERSED vs plain Conv1d's (Cout, Cin, K)), which happens to
    // be exactly the shape run_cudnn_conv1d_backward_data's `w` argument
    // expects for the equivalent "virtual" regular Conv1d whose
    // backward-data pass this call reuses as ConvTranspose1d's forward.
    int Cin_t = w->shape[0], Cout_t = w->shape[1], K = w->shape[2];
    int B = x->shape[0], L_in_x = x->shape[2];
    if (x->shape[1] != Cin_t)
        throw std::invalid_argument("conv1d_transpose_cudnn: input channels (" + std::to_string(x->shape[1]) +
                                     ") don't match weight's in_channels (" + std::to_string(Cin_t) + ")");

    auto y = run_cudnn_conv1d_backward_data(x, w, B, Cout_t, L_out, stride, padding, dilation, L_in_x);

    if (g_grad_enabled && (x->requires_grad || w->requires_grad)) {
        y->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {x, w};
        node->op_name = "conv1d_transpose_cudnn";
        int Cin_c = Cin_t, K_c = K, S_c = stride, P_c = padding, D_c = dilation, Lin_c = L_in_x;
        node->backward_fn = [x, w, Cin_c, K_c, S_c, P_c, D_c, Lin_c](std::shared_ptr<Tensor> grad_out) {
            // grad wrt x: the adjoint of backward_data is ordinary forward
            // convolution — run the virtual conv's plain forward on
            // upstream grad_out.
            auto grad_x = run_cudnn_conv1d_forward(grad_out, w, S_c, P_c, D_c, Lin_c);
            // grad wrt weight: the virtual conv's backward_filter, with
            // roles swapped — grad_out standing in as "x" and the
            // transpose's original input standing in as "grad_y".
            auto grad_w = run_cudnn_conv1d_backward_filter(grad_out, x, Cin_c, K_c, S_c, P_c, D_c, Lin_c);
            return std::vector<std::shared_ptr<Tensor>>{grad_x, grad_w};
        };
        y->grad_fn = node;
    }
    return y;
}

static std::shared_ptr<Tensor> dispatch_conv2d_cudnn(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w,
                                                       int SH, int SW, int PH, int PW, int DH, int DW) {
    if (x->device != "cuda" || w->device != "cuda")
        throw std::runtime_error("conv2d_cudnn: both input and weight must be on device='cuda'.");
    if (x->dtype != DType::FLOAT32 || w->dtype != DType::FLOAT32)
        throw std::runtime_error("conv2d_cudnn: only float32 is supported. Use conv2d_im2col (matmul-based) instead.");
    if (!x->is_contiguous()) x = dispatch_contiguous(x);
    if (!w->is_contiguous()) w = dispatch_contiguous(w);

    int B = x->shape[0], C_in = x->shape[1], H = x->shape[2], W = x->shape[3];
    int C_out = w->shape[0], KH = w->shape[2], KW = w->shape[3];
    int OH = (H + 2 * PH - DH * (KH - 1) - 1) / SH + 1;
    int OW = (W + 2 * PW - DW * (KW - 1) - 1) / SW + 1;
    if (OH <= 0 || OW <= 0)
        throw std::invalid_argument("conv2d_cudnn: computed output size <= 0.");

    auto y = run_cudnn_conv2d_forward(x, w, SH, SW, PH, PW, DH, DW, OH, OW);

    if (g_grad_enabled && (x->requires_grad || w->requires_grad)) {
        y->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {x, w};
        node->op_name = "conv2d_cudnn";
        int B_c=B, Cin_c=C_in, H_c=H, W_c=W, Cout_c=C_out, KH_c=KH, KW_c=KW,
            SH_c=SH, SW_c=SW, PH_c=PH, PW_c=PW, DH_c=DH, DW_c=DW, OH_c=OH, OW_c=OW;
        node->backward_fn = [x, w, B_c, Cin_c, H_c, W_c, Cout_c, KH_c, KW_c, SH_c, SW_c, PH_c, PW_c, DH_c, DW_c, OH_c, OW_c]
                             (std::shared_ptr<Tensor> grad_out) {
            auto grad_x = run_cudnn_conv2d_backward_data(grad_out, w, B_c, Cin_c, H_c, W_c, SH_c, SW_c, PH_c, PW_c, DH_c, DW_c, OH_c, OW_c, /*require_capturable=*/false);
            auto grad_w = run_cudnn_conv2d_backward_filter(x, grad_out, Cout_c, KH_c, KW_c, SH_c, SW_c, PH_c, PW_c, DH_c, DW_c, OH_c, OW_c, /*require_capturable=*/false);
            return std::vector<std::shared_ptr<Tensor>>{grad_x, grad_w};
        };
        y->grad_fn = node;
    }
    return y;
}

static std::shared_ptr<Tensor> dispatch_conv3d_cudnn(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w,
                                                       int SD, int SH, int SW, int PD, int PH, int PW,
                                                       int DD, int DH, int DW) {
    if (x->device != "cuda" || w->device != "cuda")
        throw std::runtime_error("conv3d_cudnn: both input and weight must be on device='cuda'.");
    if (x->dtype != DType::FLOAT32 || w->dtype != DType::FLOAT32)
        throw std::runtime_error("conv3d_cudnn: only float32 is supported. Use conv3d_im2col (matmul-based) instead.");
    if (!x->is_contiguous()) x = dispatch_contiguous(x);
    if (!w->is_contiguous()) w = dispatch_contiguous(w);

    int B = x->shape[0], Cin = x->shape[1], D = x->shape[2], H = x->shape[3], W = x->shape[4];
    int Cout = w->shape[0], KD = w->shape[2], KH = w->shape[3], KW = w->shape[4];
    int OD = (D + 2*PD - DD*(KD-1) - 1) / SD + 1;
    int OH = (H + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (W + 2*PW - DW*(KW-1) - 1) / SW + 1;
    if (OD <= 0 || OH <= 0 || OW <= 0)
        throw std::invalid_argument("conv3d_cudnn: computed output size <= 0.");

    auto y = run_cudnn_conv3d_forward(x, w, SD, SH, SW, PD, PH, PW, DD, DH, DW, OD, OH, OW);

    if (g_grad_enabled && (x->requires_grad || w->requires_grad)) {
        y->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {x, w};
        node->op_name = "conv3d_cudnn";
        int B_c=B,Cin_c=Cin,D_c=D,H_c=H,W_c=W,Cout_c=Cout,KD_c=KD,KH_c=KH,KW_c=KW,
            SD_c=SD,SH_c=SH,SW_c=SW,PD_c=PD,PH_c=PH,PW_c=PW,DD_c=DD,DH_c=DH,DW_c=DW,OD_c=OD,OH_c=OH,OW_c=OW;
        node->backward_fn = [x, w, B_c,Cin_c,D_c,H_c,W_c,Cout_c,KD_c,KH_c,KW_c,SD_c,SH_c,SW_c,PD_c,PH_c,PW_c,DD_c,DH_c,DW_c,OD_c,OH_c,OW_c]
                             (std::shared_ptr<Tensor> grad_out) {
            auto grad_x = run_cudnn_conv3d_backward_data(grad_out, w, B_c,Cin_c,D_c,H_c,W_c,SD_c,SH_c,SW_c,PD_c,PH_c,PW_c,DD_c,DH_c,DW_c,OD_c,OH_c,OW_c);
            auto grad_w = run_cudnn_conv3d_backward_filter(x, grad_out, Cout_c,KD_c,KH_c,KW_c,SD_c,SH_c,SW_c,PD_c,PH_c,PW_c,DD_c,DH_c,DW_c,OD_c,OH_c,OW_c);
            return std::vector<std::shared_ptr<Tensor>>{grad_x, grad_w};
        };
        y->grad_fn = node;
    }
    return y;
}

static std::shared_ptr<Tensor> dispatch_conv2d_transpose_cudnn(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w,
                                                                 int SH, int SW, int PH, int PW, int DH, int DW,
                                                                 int OH, int OW) {
    if (x->device != "cuda" || w->device != "cuda")
        throw std::runtime_error("conv2d_transpose_cudnn: both input and weight must be on device='cuda'.");
    if (x->dtype != DType::FLOAT32 || w->dtype != DType::FLOAT32)
        throw std::runtime_error("conv2d_transpose_cudnn: only float32 is supported.");
    if (!x->is_contiguous()) x = dispatch_contiguous(x);
    if (!w->is_contiguous()) w = dispatch_contiguous(w);

    // weight: (Cin_t, Cout_t, KH, KW) — reversed vs Conv2d's (Cout,Cin,KH,KW).
    int Cin_t = w->shape[0], Cout_t = w->shape[1], KH = w->shape[2], KW = w->shape[3];
    int B = x->shape[0], H_in_x = x->shape[2], W_in_x = x->shape[3];
    if (x->shape[1] != Cin_t)
        throw std::invalid_argument("conv2d_transpose_cudnn: input channels don't match weight's in_channels");

    auto y = run_cudnn_conv2d_backward_data(x, w, B, Cout_t, OH, OW, SH, SW, PH, PW, DH, DW,
                                             H_in_x, W_in_x, /*require_capturable=*/false);

    if (g_grad_enabled && (x->requires_grad || w->requires_grad)) {
        y->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {x, w};
        node->op_name = "conv2d_transpose_cudnn";
        int Cin_c=Cin_t, KH_c=KH, KW_c=KW, SH_c=SH, SW_c=SW, PH_c=PH, PW_c=PW, DH_c=DH, DW_c=DW, Hin_c=H_in_x, Win_c=W_in_x;
        node->backward_fn = [x, w, Cin_c,KH_c,KW_c,SH_c,SW_c,PH_c,PW_c,DH_c,DW_c,Hin_c,Win_c]
                             (std::shared_ptr<Tensor> grad_out) {
            auto grad_x = run_cudnn_conv2d_forward(grad_out, w, SH_c, SW_c, PH_c, PW_c, DH_c, DW_c, Hin_c, Win_c);
            auto grad_w = run_cudnn_conv2d_backward_filter(grad_out, x, Cin_c, KH_c, KW_c, SH_c, SW_c, PH_c, PW_c,
                                                            DH_c, DW_c, Hin_c, Win_c, /*require_capturable=*/false);
            return std::vector<std::shared_ptr<Tensor>>{grad_x, grad_w};
        };
        y->grad_fn = node;
    }
    return y;
}

static std::shared_ptr<Tensor> dispatch_conv3d_transpose_cudnn(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w,
                                                                 int SD, int SH, int SW, int PD, int PH, int PW,
                                                                 int DD, int DH, int DW, int OD, int OH, int OW) {
    if (x->device != "cuda" || w->device != "cuda")
        throw std::runtime_error("conv3d_transpose_cudnn: both input and weight must be on device='cuda'.");
    if (x->dtype != DType::FLOAT32 || w->dtype != DType::FLOAT32)
        throw std::runtime_error("conv3d_transpose_cudnn: only float32 is supported.");
    if (!x->is_contiguous()) x = dispatch_contiguous(x);
    if (!w->is_contiguous()) w = dispatch_contiguous(w);

    int Cin_t = w->shape[0], Cout_t = w->shape[1], KD = w->shape[2], KH = w->shape[3], KW = w->shape[4];
    int B = x->shape[0], D_in_x = x->shape[2], H_in_x = x->shape[3], W_in_x = x->shape[4];
    if (x->shape[1] != Cin_t)
        throw std::invalid_argument("conv3d_transpose_cudnn: input channels don't match weight's in_channels");

    auto y = run_cudnn_conv3d_backward_data(x, w, B, Cout_t, OD, OH, OW, SD, SH, SW, PD, PH, PW,
                                             DD, DH, DW, D_in_x, H_in_x, W_in_x);

    if (g_grad_enabled && (x->requires_grad || w->requires_grad)) {
        y->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {x, w};
        node->op_name = "conv3d_transpose_cudnn";
        int Cin_c=Cin_t, KD_c=KD, KH_c=KH, KW_c=KW, SD_c=SD, SH_c=SH, SW_c=SW,
            PD_c=PD, PH_c=PH, PW_c=PW, DD_c=DD, DH_c=DH, DW_c=DW, Din_c=D_in_x, Hin_c=H_in_x, Win_c=W_in_x;
        node->backward_fn = [x, w, Cin_c,KD_c,KH_c,KW_c,SD_c,SH_c,SW_c,PD_c,PH_c,PW_c,DD_c,DH_c,DW_c,Din_c,Hin_c,Win_c]
                             (std::shared_ptr<Tensor> grad_out) {
            auto grad_x = run_cudnn_conv3d_forward(grad_out, w, SD_c, SH_c, SW_c, PD_c, PH_c, PW_c,
                                                    DD_c, DH_c, DW_c, Din_c, Hin_c, Win_c);
            auto grad_w = run_cudnn_conv3d_backward_filter(grad_out, x, Cin_c, KD_c, KH_c, KW_c, SD_c, SH_c, SW_c,
                                                            PD_c, PH_c, PW_c, DD_c, DH_c, DW_c, Din_c, Hin_c, Win_c);
            return std::vector<std::shared_ptr<Tensor>>{grad_x, grad_w};
        };
        y->grad_fn = node;
    }
    return y;
}

#endif  // AAKAAR_HAS_CUDNN


// Sums `grad` down to `target_shape` following numpy/torch broadcasting-gradient
// rules: any leading dims present in `grad` but absent from `target_shape` are
// summed away entirely (they existed only because of broadcasting), and any
// aligned dim where `target_shape` is 1 but `grad` is not gets summed with
// keepdim=true (that axis was broadcast from size 1).
//
// `skip_trailing` lets matmul reuse this for its batch dims only, leaving the
// trailing (M,K)/(K,N) dims untouched — those are never broadcast targets for
// matmul, they're the actual contraction/output dims.
static std::shared_ptr<Tensor> reduce_grad_to_shape(std::shared_ptr<Tensor> grad,
                                                     const std::vector<int>& target_shape,
                                                     int skip_trailing = 0) {
    int nd_g = (int)grad->shape.size();
    int nd_t = (int)target_shape.size();

    if (nd_g < nd_t)
        throw std::runtime_error(
            "reduce_grad_to_shape: gradient has fewer dims (" + std::to_string(nd_g) +
            ") than its target shape (" + std::to_string(nd_t) +
            "). This indicates a bug upstream in the forward/backward broadcasting logic.");

    auto g = grad;
    int offset = nd_g - nd_t;
    // Leading extra dims: target never had them, so they're pure broadcast axes.
    // Always sum axis 0 repeatedly since each removal shifts everything down.
    for (int i = 0; i < offset; ++i) {
        g = dispatch_sum_axis(g, 0, false);
    }

    // Aligned dims: sum where target was 1 but the (now offset-adjusted) grad isn't.
    int nd_g_now = (int)g->shape.size();
    for (int i = 0; i < nd_t - skip_trailing; ++i) {
        if (i >= nd_g_now) break;  // defensive; shouldn't happen given the checks above
        if (target_shape[i] == 1 && g->shape[i] != 1) {
            g = dispatch_sum_axis(g, i, true);
        }
    }

    if (g->shape != target_shape && skip_trailing == 0) {
        throw std::runtime_error(
            "reduce_grad_to_shape: reduced gradient shape does not match target shape "
            "after reduction. This means the forward op's broadcasting and this backward "
            "reduction have gone out of sync — check the forward kernel's broadcast rules.");
    }
    return g;
}

static std::shared_ptr<Tensor> dispatch_add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    int C, HW;
    if (a->dtype == DType::FLOAT32 && b->dtype == DType::FLOAT32 &&
        a->is_contiguous() && b->is_contiguous() &&
        is_channel_bias_shape(b->shape, a->shape, C, HW)) {
        auto result = std::make_shared<Tensor>(a->shape, a->device);
        int total = a->size;
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") run_cuda_add_channel_bias(a, b, result, C, HW, total);
        else
#endif
        run_cpu_add_channel_bias(a, b, result, C, HW, total);

        if (g_grad_enabled && (a->requires_grad || b->requires_grad)) {
            result->requires_grad = true;
            auto node = std::make_shared<Node>();
            node->inputs = {a, b};
            node->op_name = "add_channel_bias";
            auto a_shape = a->shape;
            auto b_shape = b->shape;
            
            // Updated backward_fn for fast-path channel bias gradient
            node->backward_fn = [a_shape, b_shape, C, HW](std::shared_ptr<Tensor> grad_out) {
                auto da = reduce_grad_to_shape(grad_out, a_shape);  // no-op: shapes already match
                auto db = std::make_shared<Tensor>(b_shape, grad_out->device);
                int B = a_shape[0];
#ifndef AAKAAR_NO_CUDA
                if (grad_out->device == "cuda") run_cuda_channel_bias_grad_nd(grad_out, db, B, C, HW);
                else
#endif
                run_cpu_channel_bias_grad_nd(grad_out, db, B, C, HW);
                return std::vector<std::shared_ptr<Tensor>>{da, db};
            };
            
            result->grad_fn = node;
        }
        return result;
    }
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && (a->requires_grad || b->requires_grad))
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_add_typed(a, b);
#endif
        return run_cpu_add_typed(a, b);
    }

    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_add(a, b);
    else
#endif
    result = run_cpu_add(a, b);

    if (g_grad_enabled && (a->requires_grad || b->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->op_name = "add";
        auto a_shape = a->shape;
        auto b_shape = b->shape;
        node->backward_fn = [a_shape, b_shape](std::shared_ptr<Tensor> grad_out) {
            auto da = reduce_grad_to_shape(grad_out, a_shape);
            auto db = reduce_grad_to_shape(grad_out, b_shape);
            return std::vector<std::shared_ptr<Tensor>>{da, db};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && (a->requires_grad || b->requires_grad))
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_sub_typed(a, b);
#endif
        return run_cpu_sub_typed(a, b);
    }

    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_sub(a, b);
    else
#endif
    result = run_cpu_sub(a, b);

    if (g_grad_enabled && (a->requires_grad || b->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->op_name = "sub";
        auto a_shape = a->shape;
        auto b_shape = b->shape;
        node->backward_fn = [a_shape, b_shape](std::shared_ptr<Tensor> grad_out) {
            auto neg = dispatch_mul_scalar(grad_out, -1.0f);
            auto da = reduce_grad_to_shape(grad_out, a_shape);
            auto db = reduce_grad_to_shape(neg, b_shape);
            return std::vector<std::shared_ptr<Tensor>>{da, db};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && (a->requires_grad || b->requires_grad))
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_mul_typed(a, b);
#endif
        return run_cpu_mul_typed(a, b);
    }

    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_mul(a, b);
    else
#endif
    result = run_cpu_mul(a, b);

    if (g_grad_enabled && (a->requires_grad || b->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->op_name = "mul";
        node->backward_fn = [a, b](std::shared_ptr<Tensor> grad_out) {
            auto da_full = dispatch_mul(grad_out, b);
            auto db_full = dispatch_mul(grad_out, a);
            auto da = reduce_grad_to_shape(da_full, a->shape);
            auto db = reduce_grad_to_shape(db_full, b->shape);
            return std::vector<std::shared_ptr<Tensor>>{da, db};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_div(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && (a->requires_grad || b->requires_grad))
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_div_typed(a, b);
#endif
        return run_cpu_div_typed(a, b);
    }

    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_div(a, b);
    else
#endif
    result = run_cpu_div(a, b);

    if (g_grad_enabled && (a->requires_grad || b->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->op_name = "div";
        node->backward_fn = [a, b](std::shared_ptr<Tensor> grad_out) {
            auto da_full = dispatch_div(grad_out, b);
            auto b_sq = dispatch_mul(b, b);
            auto a_over_bsq = dispatch_div(a, b_sq);
            auto neg = dispatch_mul_scalar(a_over_bsq, -1.0f);
            auto db_full = dispatch_mul(grad_out, neg);
            auto da = reduce_grad_to_shape(da_full, a->shape);
            auto db = reduce_grad_to_shape(db_full, b->shape);
            return std::vector<std::shared_ptr<Tensor>>{da, db};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    // 1. Float64 Execution Path (No Autograd Support)
    if (a->dtype == DType::FLOAT64) {
        if (g_grad_enabled && (a->requires_grad || b->requires_grad)) {
            throw std::runtime_error("Autograd is not yet supported for dtype 'float64'. "
                                     "Only float32 currently supports gradients.");
        }
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_matmul_f64(a, b);
#endif
        return run_cpu_matmul_f64(a, b);
    }

    // 2. Updated Integer Execution Path (No Autograd Support)
    if (a->dtype == DType::INT32 || a->dtype == DType::INT64) {
        if (g_grad_enabled && (a->requires_grad || b->requires_grad)) {
            throw std::runtime_error("Autograd is not yet supported for dtype '" + 
                                     dtype_name(a->dtype) + "'. Only float32 currently supports gradients.");
        }
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_matmul_int_typed(a, b);
#endif
        return run_cpu_matmul_int_typed(a, b);
    }

    // 3. Existing Float32 Execution Path
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda")
        result = run_cublas_matmul(a, b);
    else
#endif
        result = run_cpu_matmul(a, b);

    // 4. Float32 Autograd Graph Construction
    if (g_grad_enabled && (a->requires_grad || b->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a, b};
        node->op_name = "matmul";
        node->backward_fn = [a, b](std::shared_ptr<Tensor> grad_out) {
            // Fast path: both operands genuinely 2D (the common case — Linear
            // layers). Skips the transpose+contiguous materialization entirely,
            // passing cuBLAS/BLAS transpose flags against the original memory
            // instead. Batched (>2D) matmul falls back to the general path below,
            // since the transpose-flag trick doesn't extend cleanly to a batched
            // dimension without per-batch striding cuBLAS doesn't expose simply.
            bool is_2d = a->shape.size() == 2 && b->shape.size() == 2 && grad_out->shape.size() == 2;
            if (is_2d && a->dtype == DType::FLOAT32) {
                std::shared_ptr<Tensor> da, db;
        #ifndef AAKAAR_NO_CUDA
                if (a->device == "cuda") {
                    da = run_cublas_matmul_a_bt(grad_out, b);
                    db = run_cublas_matmul_at_b(a, grad_out);
                } else
        #endif
                {
                    da = run_cpu_matmul_a_bt(grad_out, b);
                    db = run_cpu_matmul_at_b(a, grad_out);
                }
                return std::vector<std::shared_ptr<Tensor>>{da, db};
            }

            // General N-D fallback (unchanged)
            int nd_a = (int)a->shape.size();
            int nd_b = (int)b->shape.size();
            auto bT = b->transpose(nd_b-2, nd_b-1)->contiguous();
            auto aT = a->transpose(nd_a-2, nd_a-1)->contiguous();
            auto da_full = dispatch_matmul(grad_out, bT);
            auto db_full = dispatch_matmul(aT, grad_out);
            auto da = reduce_grad_to_shape(da_full, a->shape, 2);
            auto db = reduce_grad_to_shape(db_full, b->shape, 2);
            return std::vector<std::shared_ptr<Tensor>>{da, db};
        };
        result->grad_fn = node;
    }

    return result;
}

static std::shared_ptr<Tensor> dispatch_sign(std::shared_ptr<Tensor> a) {
    if (a->dtype != DType::FLOAT32) {
        if (g_grad_enabled && a->requires_grad)
            throw std::runtime_error("Autograd is not yet supported for dtype '" + dtype_name(a->dtype) +
                                      "'. Only float32 currently supports gradients.");
        if (!a->is_contiguous()) a = dispatch_contiguous(a);
#ifndef AAKAAR_NO_CUDA
        if (a->device == "cuda") return run_cuda_sign_typed(a);
#endif
        return run_cpu_sign_typed(a);
    }
    if (!a->is_contiguous()) a = dispatch_contiguous(a);
    std::shared_ptr<Tensor> result;
#ifndef AAKAAR_NO_CUDA
    if (a->device == "cuda") result = run_cuda_sign(a);
    else
#endif
    result = run_cpu_sign(a);

    if (g_grad_enabled && a->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {a};
        node->op_name = "sign";
        node->backward_fn = [](std::shared_ptr<Tensor> grad_out) {
            // sign's derivative is zero almost everywhere — matches torch's
            // convention of returning zero gradient, not an error.
            auto zero_grad = dispatch_mul_scalar(grad_out, 0.0f);
            return std::vector<std::shared_ptr<Tensor>>{zero_grad};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_huber_loss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float delta) {
    if (pred->dtype != DType::FLOAT32 || target->dtype != DType::FLOAT32)
        throw std::runtime_error("huber_loss(): only float32 is currently supported by the fused kernel.");
    if (!pred->is_contiguous()) pred = dispatch_contiguous(pred);
    if (!target->is_contiguous()) target = dispatch_contiguous(target);

    float total;
#ifndef AAKAAR_NO_CUDA
    if (pred->device == "cuda") {
        auto r = run_cuda_huber_loss_forward(pred, target, delta);
        total = r.first;
    } else
#endif
    total = run_cpu_huber_loss_forward(pred, target, delta);

    int n = pred->size;
    float mean_loss = total / n;

    auto result = std::make_shared<Tensor>(std::vector<int>{1}, pred->device);
#ifndef AAKAAR_NO_CUDA
    if (pred->device == "cuda") cudaMemcpy(result->fptr(), &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
    else
#endif
    result->fptr()[0] = mean_loss;

    if (g_grad_enabled && (pred->requires_grad || target->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {pred, target};
        node->op_name = "huber_loss";
        node->backward_fn = [pred, target, delta, n](std::shared_ptr<Tensor> grad_out) {
            float grad_out_scalar;
#ifndef AAKAAR_NO_CUDA
            if (grad_out->device == "cuda") cudaMemcpy(&grad_out_scalar, grad_out->fptr(), sizeof(float), cudaMemcpyDeviceToHost);
            else
#endif
            grad_out_scalar = grad_out->fptr()[0];

            float grad_scale = grad_out_scalar / n;
            std::shared_ptr<Tensor> grad_pred, grad_target;
#ifndef AAKAAR_NO_CUDA
            if (pred->device == "cuda") {
                grad_pred = run_cuda_huber_loss_backward(pred, target, delta, grad_scale);
                grad_target = run_cuda_huber_loss_backward(pred, target, delta, -grad_scale);
            } else
#endif
            {
                grad_pred = run_cpu_huber_loss_backward(pred, target, delta, grad_scale);
                grad_target = run_cpu_huber_loss_backward(pred, target, delta, -grad_scale);
            }
            return std::vector<std::shared_ptr<Tensor>>{grad_pred, grad_target};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_cross_entropy_fused(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> onehot) {
    if (logits->dtype != DType::FLOAT32 || onehot->dtype != DType::FLOAT32)
        throw std::runtime_error("cross_entropy(): only float32 is currently supported by the fused kernel.");
    if (logits->shape.size() != 2)
        throw std::runtime_error("cross_entropy(): fused kernel currently requires 2D (batch, classes) input.");
    if (!logits->is_contiguous()) logits = dispatch_contiguous(logits);
    if (!onehot->is_contiguous()) onehot = dispatch_contiguous(onehot);

    std::shared_ptr<Tensor> per_row;
#ifndef AAKAAR_NO_CUDA
    if (logits->device == "cuda") per_row = run_cuda_cross_entropy_per_row(logits, onehot);
    else
#endif
    per_row = run_cpu_cross_entropy_per_row(logits, onehot);

    int N = logits->shape[0];
    float total = 0.0f;
    auto per_row_np = per_row->fptr();
#ifndef AAKAAR_NO_CUDA
    if (logits->device == "cuda") {
        std::vector<float> host_buf(N);
        cudaMemcpy(host_buf.data(), per_row->fptr(), N * sizeof(float), cudaMemcpyDeviceToHost);
        for (float v : host_buf) total += v;
    } else
#endif
    { for (int i = 0; i < N; ++i) total += per_row_np[i]; }

    float mean_loss = total / N;
    auto result = std::make_shared<Tensor>(std::vector<int>{1}, logits->device);
#ifndef AAKAAR_NO_CUDA
    if (logits->device == "cuda") cudaMemcpy(result->fptr(), &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
    else
#endif
    result->fptr()[0] = mean_loss;

    if (g_grad_enabled && (logits->requires_grad || onehot->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {logits, onehot};
        node->op_name = "cross_entropy_fused";
        node->backward_fn = [logits, onehot, N](std::shared_ptr<Tensor> grad_out) {
            float grad_out_scalar;
#ifndef AAKAAR_NO_CUDA
            if (grad_out->device == "cuda") cudaMemcpy(&grad_out_scalar, grad_out->fptr(), sizeof(float), cudaMemcpyDeviceToHost);
            else
#endif
            grad_out_scalar = grad_out->fptr()[0];
            float grad_scale = grad_out_scalar / N;

            std::shared_ptr<Tensor> grad_logits;
#ifndef AAKAAR_NO_CUDA
            if (logits->device == "cuda") grad_logits = run_cuda_cross_entropy_backward(logits, onehot, grad_scale);
            else
#endif
            grad_logits = run_cpu_cross_entropy_backward(logits, onehot, grad_scale);

            // No meaningful gradient w.r.t. the one-hot target in this
            // formulation (it's a fixed label, not a learned input) —
            // return a zero tensor of matching shape for it.
            auto zero_target_grad = std::make_shared<Tensor>(onehot->shape, onehot->device);
            zero_target_grad->fill_zero();
            return std::vector<std::shared_ptr<Tensor>>{grad_logits, zero_target_grad};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_bce_fused(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target) {
    if (pred->dtype != DType::FLOAT32 || target->dtype != DType::FLOAT32)
        throw std::runtime_error("BCELoss(): only float32 is currently supported by the fused kernel.");
    if (!pred->is_contiguous()) pred = dispatch_contiguous(pred);
    if (!target->is_contiguous()) target = dispatch_contiguous(target);

    float total; int n;
#ifndef AAKAAR_NO_CUDA
    if (pred->device == "cuda") { auto r = run_cuda_bce_forward(pred, target); total = r.first; n = r.second; }
    else
#endif
    { auto r = run_cpu_bce_forward(pred, target); total = r.first; n = r.second; }

    float mean_loss = total / n;
    auto result = std::make_shared<Tensor>(std::vector<int>{1}, pred->device);
#ifndef AAKAAR_NO_CUDA
    if (pred->device == "cuda") cudaMemcpy(result->fptr(), &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
    else
#endif
    result->fptr()[0] = mean_loss;

    if (g_grad_enabled && (pred->requires_grad || target->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {pred, target};
        node->op_name = "bce_fused";
        node->backward_fn = [pred, target, n](std::shared_ptr<Tensor> grad_out) {
            float g;
#ifndef AAKAAR_NO_CUDA
            if (grad_out->device == "cuda") cudaMemcpy(&g, grad_out->fptr(), sizeof(float), cudaMemcpyDeviceToHost);
            else
#endif
            g = grad_out->fptr()[0];
            float grad_scale = g / n;

            std::shared_ptr<Tensor> grad_pred, grad_target;
#ifndef AAKAAR_NO_CUDA
            if (pred->device == "cuda") {
                grad_pred = run_cuda_bce_backward(pred, target, grad_scale);
                grad_target = run_cuda_bce_backward(pred, target, -grad_scale);
            } else
#endif
            {
                grad_pred = run_cpu_bce_backward(pred, target, grad_scale);
                grad_target = run_cpu_bce_backward(pred, target, -grad_scale);
            }
            return std::vector<std::shared_ptr<Tensor>>{grad_pred, grad_target};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_smoothl1_fused(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target, float beta) {
    if (pred->dtype != DType::FLOAT32 || target->dtype != DType::FLOAT32)
        throw std::runtime_error("SmoothL1Loss(): only float32 is currently supported by the fused kernel.");
    if (!pred->is_contiguous()) pred = dispatch_contiguous(pred);
    if (!target->is_contiguous()) target = dispatch_contiguous(target);

    float total; int n;
#ifndef AAKAAR_NO_CUDA
    if (pred->device == "cuda") { auto r = run_cuda_smoothl1_forward(pred, target, beta); total = r.first; n = r.second; }
    else
#endif
    { auto r = run_cpu_smoothl1_forward(pred, target, beta); total = r.first; n = r.second; }

    float mean_loss = total / n;
    auto result = std::make_shared<Tensor>(std::vector<int>{1}, pred->device);
#ifndef AAKAAR_NO_CUDA
    if (pred->device == "cuda") cudaMemcpy(result->fptr(), &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
    else
#endif
    result->fptr()[0] = mean_loss;

    if (g_grad_enabled && (pred->requires_grad || target->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {pred, target};
        node->op_name = "smoothl1_fused";
        node->backward_fn = [pred, target, beta, n](std::shared_ptr<Tensor> grad_out) {
            float g;
#ifndef AAKAAR_NO_CUDA
            if (grad_out->device == "cuda") cudaMemcpy(&g, grad_out->fptr(), sizeof(float), cudaMemcpyDeviceToHost);
            else
#endif
            g = grad_out->fptr()[0];
            float grad_scale = g / n;

            std::shared_ptr<Tensor> grad_pred, grad_target;
#ifndef AAKAAR_NO_CUDA
            if (pred->device == "cuda") {
                grad_pred = run_cuda_smoothl1_backward(pred, target, beta, grad_scale);
                grad_target = run_cuda_smoothl1_backward(pred, target, beta, -grad_scale);
            } else
#endif
            {
                grad_pred = run_cpu_smoothl1_backward(pred, target, beta, grad_scale);
                grad_target = run_cpu_smoothl1_backward(pred, target, beta, -grad_scale);
            }
            return std::vector<std::shared_ptr<Tensor>>{grad_pred, grad_target};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_multimargin_fused(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> onehot, float margin) {
    if (logits->dtype != DType::FLOAT32 || onehot->dtype != DType::FLOAT32)
        throw std::runtime_error("MultiMarginLoss(): only float32 is currently supported by the fused kernel.");
    if (logits->shape.size() != 2)
        throw std::runtime_error("MultiMarginLoss(): fused kernel currently requires 2D (batch, classes) input.");
    if (!logits->is_contiguous()) logits = dispatch_contiguous(logits);
    if (!onehot->is_contiguous()) onehot = dispatch_contiguous(onehot);

    std::shared_ptr<Tensor> per_row;
#ifndef AAKAAR_NO_CUDA
    if (logits->device == "cuda") per_row = run_cuda_multimargin_per_row(logits, onehot, margin);
    else
#endif
    per_row = run_cpu_multimargin_per_row(logits, onehot, margin);

    int N = logits->shape[0];
    float total = 0.0f;
#ifndef AAKAAR_NO_CUDA
    if (logits->device == "cuda") {
        std::vector<float> host_buf(N);
        cudaMemcpy(host_buf.data(), per_row->fptr(), N * sizeof(float), cudaMemcpyDeviceToHost);
        for (float v : host_buf) total += v;
    } else
#endif
    { auto pr = per_row->fptr(); for (int i = 0; i < N; ++i) total += pr[i]; }

    float mean_loss = total / N;
    auto result = std::make_shared<Tensor>(std::vector<int>{1}, logits->device);
#ifndef AAKAAR_NO_CUDA
    if (logits->device == "cuda") cudaMemcpy(result->fptr(), &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
    else
#endif
    result->fptr()[0] = mean_loss;

    if (g_grad_enabled && (logits->requires_grad || onehot->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {logits, onehot};
        node->op_name = "multimargin_fused";
        node->backward_fn = [logits, onehot, margin, N](std::shared_ptr<Tensor> grad_out) {
            float g;
#ifndef AAKAAR_NO_CUDA
            if (grad_out->device == "cuda") cudaMemcpy(&g, grad_out->fptr(), sizeof(float), cudaMemcpyDeviceToHost);
            else
#endif
            g = grad_out->fptr()[0];
            float grad_scale = g / N;

            std::shared_ptr<Tensor> grad_logits;
#ifndef AAKAAR_NO_CUDA
            if (logits->device == "cuda") grad_logits = run_cuda_multimargin_backward(logits, onehot, margin, grad_scale);
            else
#endif
            grad_logits = run_cpu_multimargin_backward(logits, onehot, margin, grad_scale);

            auto zero_target_grad = std::make_shared<Tensor>(onehot->shape, onehot->device);
            zero_target_grad->fill_zero();
            return std::vector<std::shared_ptr<Tensor>>{grad_logits, zero_target_grad};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_mse_fused(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target) {
    if (pred->dtype != DType::FLOAT32 || target->dtype != DType::FLOAT32)
        throw std::runtime_error("MSELoss(): only float32 is currently supported by the fused kernel.");
    if (!pred->is_contiguous()) pred = dispatch_contiguous(pred);
    if (!target->is_contiguous()) target = dispatch_contiguous(target);

    float total; int n;
#ifndef AAKAAR_NO_CUDA
    if (pred->device == "cuda") { auto r = run_cuda_mse_forward(pred, target); total = r.first; n = r.second; }
    else
#endif
    { auto r = run_cpu_mse_forward(pred, target); total = r.first; n = r.second; }

    float mean_loss = total / n;
    auto result = std::make_shared<Tensor>(std::vector<int>{1}, pred->device);
#ifndef AAKAAR_NO_CUDA
    if (pred->device == "cuda") cudaMemcpy(result->fptr(), &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
    else
#endif
    result->fptr()[0] = mean_loss;

    if (g_grad_enabled && (pred->requires_grad || target->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {pred, target};
        node->op_name = "mse_fused";
        node->backward_fn = [pred, target, n](std::shared_ptr<Tensor> grad_out) {
            float g;
#ifndef AAKAAR_NO_CUDA
            if (grad_out->device == "cuda") cudaMemcpy(&g, grad_out->fptr(), sizeof(float), cudaMemcpyDeviceToHost);
            else
#endif
            g = grad_out->fptr()[0];
            float grad_scale = g / n;

            std::shared_ptr<Tensor> grad_pred, grad_target;
#ifndef AAKAAR_NO_CUDA
            if (pred->device == "cuda") {
                grad_pred = run_cuda_mse_backward(pred, target, grad_scale);
                grad_target = run_cuda_mse_backward(pred, target, -grad_scale);
            } else
#endif
            {
                grad_pred = run_cpu_mse_backward(pred, target, grad_scale);
                grad_target = run_cpu_mse_backward(pred, target, -grad_scale);
            }
            return std::vector<std::shared_ptr<Tensor>>{grad_pred, grad_target};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_nll_fused(std::shared_ptr<Tensor> log_probs, std::shared_ptr<Tensor> onehot) {
    if (log_probs->dtype != DType::FLOAT32 || onehot->dtype != DType::FLOAT32)
        throw std::runtime_error("NLLLoss(): only float32 is currently supported by the fused kernel.");
    if (log_probs->shape.size() != 2)
        throw std::runtime_error("NLLLoss(): fused kernel currently requires 2D (batch, classes) input.");
    if (!log_probs->is_contiguous()) log_probs = dispatch_contiguous(log_probs);
    if (!onehot->is_contiguous()) onehot = dispatch_contiguous(onehot);

    std::shared_ptr<Tensor> per_row;
#ifndef AAKAAR_NO_CUDA
    if (log_probs->device == "cuda") per_row = run_cuda_nll_per_row(log_probs, onehot);
    else
#endif
    per_row = run_cpu_nll_per_row(log_probs, onehot);

    int N = log_probs->shape[0];
    float total = 0.0f;
#ifndef AAKAAR_NO_CUDA
    if (log_probs->device == "cuda") {
        std::vector<float> host_buf(N);
        cudaMemcpy(host_buf.data(), per_row->fptr(), N * sizeof(float), cudaMemcpyDeviceToHost);
        for (float v : host_buf) total += v;
    } else
#endif
    { auto pr = per_row->fptr(); for (int i = 0; i < N; ++i) total += pr[i]; }

    float mean_loss = total / N;
    auto result = std::make_shared<Tensor>(std::vector<int>{1}, log_probs->device);
#ifndef AAKAAR_NO_CUDA
    if (log_probs->device == "cuda") cudaMemcpy(result->fptr(), &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
    else
#endif
    result->fptr()[0] = mean_loss;

    if (g_grad_enabled && (log_probs->requires_grad || onehot->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {log_probs, onehot};
        node->op_name = "nll_fused";
        node->backward_fn = [onehot, N](std::shared_ptr<Tensor> grad_out) {
            float g;
#ifndef AAKAAR_NO_CUDA
            if (grad_out->device == "cuda") cudaMemcpy(&g, grad_out->fptr(), sizeof(float), cudaMemcpyDeviceToHost);
            else
#endif
            g = grad_out->fptr()[0];
            float grad_scale = g / N;

            std::shared_ptr<Tensor> grad_log_probs;
#ifndef AAKAAR_NO_CUDA
            if (onehot->device == "cuda") grad_log_probs = run_cuda_nll_backward(onehot, grad_scale);
            else
#endif
            grad_log_probs = run_cpu_nll_backward(onehot, grad_scale);

            auto zero_target_grad = std::make_shared<Tensor>(onehot->shape, onehot->device);
            zero_target_grad->fill_zero();
            return std::vector<std::shared_ptr<Tensor>>{grad_log_probs, zero_target_grad};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> dispatch_bce_logits_fused(std::shared_ptr<Tensor> logits, std::shared_ptr<Tensor> target) {
    if (logits->dtype != DType::FLOAT32 || target->dtype != DType::FLOAT32)
        throw std::runtime_error("BCEWithLogitsLoss(): only float32 is currently supported by the fused kernel.");
    if (!logits->is_contiguous()) logits = dispatch_contiguous(logits);
    if (!target->is_contiguous()) target = dispatch_contiguous(target);

    float total; int n;
#ifndef AAKAAR_NO_CUDA
    if (logits->device == "cuda") { auto r = run_cuda_bce_logits_forward(logits, target); total = r.first; n = r.second; }
    else
#endif
    { auto r = run_cpu_bce_logits_forward(logits, target); total = r.first; n = r.second; }

    float mean_loss = total / n;
    auto result = std::make_shared<Tensor>(std::vector<int>{1}, logits->device);
#ifndef AAKAAR_NO_CUDA
    if (logits->device == "cuda") cudaMemcpy(result->fptr(), &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
    else
#endif
    result->fptr()[0] = mean_loss;

    if (g_grad_enabled && (logits->requires_grad || target->requires_grad)) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {logits, target};
        node->op_name = "bce_logits_fused";
        node->backward_fn = [logits, target, n](std::shared_ptr<Tensor> grad_out) {
            float g;
#ifndef AAKAAR_NO_CUDA
            if (grad_out->device == "cuda") cudaMemcpy(&g, grad_out->fptr(), sizeof(float), cudaMemcpyDeviceToHost);
            else
#endif
            g = grad_out->fptr()[0];
            float grad_scale = g / n;

            std::shared_ptr<Tensor> grad_logits, grad_target;
#ifndef AAKAAR_NO_CUDA
            if (logits->device == "cuda") {
                grad_logits = run_cuda_bce_logits_backward(logits, target, grad_scale);
            } else
#endif
            {
                grad_logits = run_cpu_bce_logits_backward(logits, target, grad_scale);
            }
            // No meaningful gradient w.r.t. target (fixed label).
            auto zero_target_grad = std::make_shared<Tensor>(target->shape, target->device);
            zero_target_grad->fill_zero();
            return std::vector<std::shared_ptr<Tensor>>{grad_logits, zero_target_grad};
        };
        result->grad_fn = node;
    }
    return result;
}

static void dispatch_adam_step_fused(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad,
                                      std::shared_ptr<Tensor> m, std::shared_ptr<Tensor> v,
                                      float lr, float beta1, float beta2, float eps,
                                      float weight_decay, int t) {
    if (p->dtype != DType::FLOAT32 || grad->dtype != DType::FLOAT32 ||
        m->dtype != DType::FLOAT32 || v->dtype != DType::FLOAT32)
        throw std::runtime_error("adam_step_fused: only float32 is currently supported by the fused kernel. "
                                  "Use the generic (unfused) optimizer path for other dtypes.");
    if (p->device != grad->device || p->device != m->device || p->device != v->device)
        throw std::runtime_error("adam_step_fused: p/grad/m/v must all be on the same device.");

#ifndef AAKAAR_NO_CUDA
    if (p->device == "cuda") {
        run_cuda_adam_step(p, grad, m, v, lr, beta1, beta2, eps, weight_decay, t);
        return;
    }
#endif
    run_cpu_adam_step(p, grad, m, v, lr, beta1, beta2, eps, weight_decay, t);
}

static void dispatch_sgd_step_fused(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad, std::shared_ptr<Tensor> velocity,
                                     float lr, float momentum, float weight_decay, float dampening,
                                     int nesterov, int has_momentum, int velocity_initialized) {
#ifndef AAKAAR_NO_CUDA
    if (p->device == "cuda") { run_cuda_sgd_step(p, grad, velocity, lr, momentum, weight_decay, dampening, nesterov, has_momentum, velocity_initialized); return; }
#endif
    run_cpu_sgd_step(p, grad, velocity, lr, momentum, weight_decay, dampening, nesterov, has_momentum, velocity_initialized);
}

static void dispatch_adamw_step_fused(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad, std::shared_ptr<Tensor> m, std::shared_ptr<Tensor> v,
                                       float lr, float beta1, float beta2, float eps, float weight_decay, int t) {
#ifndef AAKAAR_NO_CUDA
    if (p->device == "cuda") { run_cuda_adamw_step(p, grad, m, v, lr, beta1, beta2, eps, weight_decay, t); return; }
#endif
    run_cpu_adamw_step(p, grad, m, v, lr, beta1, beta2, eps, weight_decay, t);
}

static void dispatch_adamax_step_fused(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad, std::shared_ptr<Tensor> m, std::shared_ptr<Tensor> u,
                                        float lr, float beta1, float beta2, float eps, float weight_decay, int t) {
#ifndef AAKAAR_NO_CUDA
    if (p->device == "cuda") { run_cuda_adamax_step(p, grad, m, u, lr, beta1, beta2, eps, weight_decay, t); return; }
#endif
    run_cpu_adamax_step(p, grad, m, u, lr, beta1, beta2, eps, weight_decay, t);
}

static void dispatch_nadam_step_fused(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad, std::shared_ptr<Tensor> m, std::shared_ptr<Tensor> v,
                                       float lr, float beta1, float beta2, float eps, float weight_decay,
                                       float mu_t, float mu_t1, float mu_product, float mu_product_next, int t) {
#ifndef AAKAAR_NO_CUDA
    if (p->device == "cuda") { run_cuda_nadam_step(p, grad, m, v, lr, beta1, beta2, eps, weight_decay, mu_t, mu_t1, mu_product, mu_product_next, t); return; }
#endif
    run_cpu_nadam_step(p, grad, m, v, lr, beta1, beta2, eps, weight_decay, mu_t, mu_t1, mu_product, mu_product_next, t);
}

static void dispatch_radam_step_fused(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad, std::shared_ptr<Tensor> m, std::shared_ptr<Tensor> v,
                                       float lr, float beta1, float beta2, float eps, float weight_decay,
                                       int use_adaptive, float r_t, int t) {
#ifndef AAKAAR_NO_CUDA
    if (p->device == "cuda") { run_cuda_radam_step(p, grad, m, v, lr, beta1, beta2, eps, weight_decay, use_adaptive, r_t, t); return; }
#endif
    run_cpu_radam_step(p, grad, m, v, lr, beta1, beta2, eps, weight_decay, use_adaptive, r_t, t);
}

static void dispatch_rmsprop_step_fused(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad, std::shared_ptr<Tensor> sq_avg, std::shared_ptr<Tensor> buf,
                                         float lr, float alpha, float eps, float weight_decay, float momentum,
                                         int has_momentum, int buf_initialized) {
#ifndef AAKAAR_NO_CUDA
    if (p->device == "cuda") { run_cuda_rmsprop_step(p, grad, sq_avg, buf, lr, alpha, eps, weight_decay, momentum, has_momentum, buf_initialized); return; }
#endif
    run_cpu_rmsprop_step(p, grad, sq_avg, buf, lr, alpha, eps, weight_decay, momentum, has_momentum, buf_initialized);
}

static void dispatch_adadelta_step_fused(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad, std::shared_ptr<Tensor> sq_avg, std::shared_ptr<Tensor> acc_delta,
                                          float lr, float rho, float eps, float weight_decay) {
#ifndef AAKAAR_NO_CUDA
    if (p->device == "cuda") { run_cuda_adadelta_step(p, grad, sq_avg, acc_delta, lr, rho, eps, weight_decay); return; }
#endif
    run_cpu_adadelta_step(p, grad, sq_avg, acc_delta, lr, rho, eps, weight_decay);
}

static void dispatch_rprop_step_fused(std::shared_ptr<Tensor> p, std::shared_ptr<Tensor> grad, std::shared_ptr<Tensor> prev_grad, std::shared_ptr<Tensor> step_size,
                                       float lr, float eta_minus, float eta_plus, float step_min, float step_max, int first_step) {
#ifndef AAKAAR_NO_CUDA
    if (p->device == "cuda") { run_cuda_rprop_step(p, grad, prev_grad, step_size, lr, eta_minus, eta_plus, step_min, step_max, first_step); return; }
#endif
    run_cpu_rprop_step(p, grad, prev_grad, step_size, lr, eta_minus, eta_plus, step_min, step_max, first_step);
}

static std::shared_ptr<Tensor> dispatch_im2col_3d(std::shared_ptr<Tensor> x, int KD, int KH, int KW,
                                                    int SD, int SH, int SW, int PD, int PH, int PW,
                                                    int DD, int DH, int DW, int OD, int OH, int OW) {
    if (x->shape.size() != 5)
        throw std::invalid_argument("im2col_3d: expected input of shape (batch, channels, D, H, W), got rank " +
                                     std::to_string(x->shape.size()));
    if (x->dtype != DType::FLOAT32)
        throw std::runtime_error("im2col_3d: only float32 is currently supported.");
    if (!x->is_contiguous()) x = dispatch_contiguous(x);

    int B = x->shape[0], C = x->shape[1];
    std::vector<int> col_shape = {B, C * KD * KH * KW, OD * OH * OW};
    auto result = std::make_shared<Tensor>(col_shape, x->device);
#ifndef AAKAAR_NO_CUDA
    if (x->device == "cuda")
        run_cuda_im2col_3d(x, result, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, OD, OH, OW);
    else
#endif
    run_cpu_im2col_3d(x, result, KD, KH, KW, SD, SH, SW, PD, PH, PW, DD, DH, DW, OD, OH, OW);

    if (g_grad_enabled && x->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {x};
        node->op_name = "im2col_3d";
        int B_c = B, C_c = C, D_c = x->shape[2], H_c = x->shape[3], W_c = x->shape[4];
        int KD_c=KD,KH_c=KH,KW_c=KW,SD_c=SD,SH_c=SH,SW_c=SW,PD_c=PD,PH_c=PH,PW_c=PW,DD_c=DD,DH_c=DH,DW_c=DW,OD_c=OD,OH_c=OH,OW_c=OW;
        auto dev = x->device;
        node->backward_fn = [B_c,C_c,D_c,H_c,W_c,KD_c,KH_c,KW_c,SD_c,SH_c,SW_c,PD_c,PH_c,PW_c,DD_c,DH_c,DW_c,OD_c,OH_c,OW_c,dev]
                             (std::shared_ptr<Tensor> grad_out) {
            auto grad_x = std::make_shared<Tensor>(std::vector<int>{B_c, C_c, D_c, H_c, W_c}, dev);
#ifndef AAKAAR_NO_CUDA
            if (dev == "cuda")
                run_cuda_col2im_3d(grad_out, grad_x, B_c,C_c,D_c,H_c,W_c,KD_c,KH_c,KW_c,SD_c,SH_c,SW_c,PD_c,PH_c,PW_c,DD_c,DH_c,DW_c,OD_c,OH_c,OW_c);
            else
#endif
            run_cpu_col2im_3d(grad_out, grad_x, B_c,C_c,D_c,H_c,W_c,KD_c,KH_c,KW_c,SD_c,SH_c,SW_c,PD_c,PH_c,PW_c,DD_c,DH_c,DW_c,OD_c,OH_c,OW_c);
            return std::vector<std::shared_ptr<Tensor>>{grad_x};
        };
        result->grad_fn = node;
    }
    return result;
}

static std::shared_ptr<Tensor> tensor_from_numpy_typed(py::array arr, std::string device, bool requires_grad) {
    py::buffer_info buf = arr.request();
    std::vector<int> shape;
    for (auto d : buf.shape) shape.push_back((int)d);
    if (shape.empty()) shape.push_back(1);
    py::dtype np_dtype = arr.dtype();
    char kind = np_dtype.kind();
    py::ssize_t itemsize = np_dtype.itemsize();

    DType dt;
    if (kind == 'f' && itemsize == 4) dt = DType::FLOAT32;
    else if (kind == 'f' && itemsize == 8) dt = DType::FLOAT64;
    else if (kind == 'i' && itemsize == 4) dt = DType::INT32;
    else if (kind == 'i' && itemsize == 8) dt = DType::INT64;
    else throw std::runtime_error("from_numpy(): unsupported numpy dtype (kind='" +
                                   std::string(1, kind) + "', itemsize=" + std::to_string(itemsize) +
                                   "). Supported: float32, float64, int32, int64.");

    auto result = std::make_shared<Tensor>(shape, device, dt);
    size_t bytes = (size_t)result->size * dtype_size(dt);

    py::array contiguous_arr = py::array::ensure(arr, py::array::c_style | py::array::forcecast);
    py::buffer_info cbuf = contiguous_arr.request();

#ifndef AAKAAR_NO_CUDA
    if (device == "cuda") {
        // Route through a reusable pinned staging buffer + a genuinely
        // async copy, instead of a synchronous cudaMemcpy straight from
        // pageable numpy memory. A pageable-source cudaMemcpy forces the
        // CUDA driver to silently allocate its own temporary pinned
        // buffer, stage into it, DMA, then tear it down — every single
        // call. On Windows/WDDM that hidden per-call allocation, not the
        // DMA itself, is what dominates. Pinning a persistent, size-keyed,
        // reused buffer removes that cost and lets this call return before
        // the transfer completes, so subsequent same-stream work (kernel
        // launches, which all run on the default stream here) overlaps
        // with it instead of stalling behind it.
        auto& pinned = PinnedAllocator::get_instance();
        auto pbuf = pinned.acquire(bytes);
        std::memcpy(pbuf.ptr, cbuf.ptr, bytes);  // host-to-host; microseconds even at MB scale
        cudaMemcpyAsync(result->data_ptr, pbuf.ptr, bytes, cudaMemcpyHostToDevice, 0);
        pinned.release(bytes, pbuf, 0);
        result->requires_grad = requires_grad;
        return result;
    }
#endif
    std::memcpy(result->data_ptr, cbuf.ptr, bytes);
    result->requires_grad = requires_grad;
    return result;
}
// ---- Module definition ----

PYBIND11_MODULE(_C, m) {
    py::enum_<DType>(m, "DType")
        .value("float32", DType::FLOAT32)
        .value("float64", DType::FLOAT64)
        .value("int32", DType::INT32)
        .value("int64", DType::INT64);

    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<std::vector<int>, std::string>())
        .def(py::init<std::vector<int>, std::string, DType>())  
        .def_property_readonly("dtype", [](Tensor &t) { return dtype_name(t.dtype); })
        .def_readonly("device", &Tensor::device)
        .def_readonly("size", &Tensor::size)
        .def_readonly("shape", &Tensor::shape)
        .def_readonly("strides", &Tensor::strides)
        .def_readwrite("requires_grad", &Tensor::requires_grad)
        .def_property("grad",
            [](Tensor &t) { return t.grad; },
            [](Tensor &t, std::shared_ptr<Tensor> g) { t.grad = g; }
        )
        .def("to_numpy", &Tensor::to_numpy)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("relu", [](std::shared_ptr<Tensor> self) { return dispatch_relu(self); })
        .def("sigmoid", [](std::shared_ptr<Tensor> self) { return dispatch_sigmoid(self); })
        .def("tanh", [](std::shared_ptr<Tensor> self) { return dispatch_tanh(self); })
        .def("copy_", &Tensor::copy_)
        .def("leaky_relu", [](std::shared_ptr<Tensor> self, double slope) { return dispatch_leaky_relu(self, slope); }, py::arg("slope") = 0.01)
        .def("max", [](std::shared_ptr<Tensor> self, int dim, bool keepdim) {
            return dispatch_max_axis(self, dim, keepdim);
        }, py::arg("dim"), py::arg("keepdim") = false)
        .def("__neg__", [](std::shared_ptr<Tensor> a) { return dispatch_mul_scalar(a, -1.0f); })
        .def("contiguous", [](std::shared_ptr<Tensor> self) { return dispatch_contiguous(self); })
        .def("to", [](std::shared_ptr<Tensor> self, std::string target_device) {
            auto result = self->to_device(target_device);
            if (g_grad_enabled && self->requires_grad && result.get() != self.get()) {
                result->requires_grad = true;
                auto node = std::make_shared<Node>();
                node->inputs = {self};
                node->op_name = "to_device";
                auto origin_device = self->device;
                node->backward_fn = [origin_device](std::shared_ptr<Tensor> grad_out) {
                    auto grad_input = grad_out->to_device(origin_device);
                    return std::vector<std::shared_ptr<Tensor>>{grad_input};
                };
                result->grad_fn = node;
            }
            return result;
        }, py::arg("target_device"))
        .def("to_device", [](std::shared_ptr<Tensor> self, std::string target_device) {
            auto result = self->to_device(target_device);
            if (g_grad_enabled && self->requires_grad && result.get() != self.get()) {
                result->requires_grad = true;
                auto node = std::make_shared<Node>();
                node->inputs = {self};
                node->op_name = "to_device";
                auto origin_device = self->device;
                node->backward_fn = [origin_device](std::shared_ptr<Tensor> grad_out) {
                    auto grad_input = grad_out->to_device(origin_device);
                    return std::vector<std::shared_ptr<Tensor>>{grad_input};
                };
                result->grad_fn = node;
            }
            return result;
        }, py::arg("target_device"))
        .def("transpose", [](std::shared_ptr<Tensor> self, int dim0, int dim1) {
            auto result = self->transpose(dim0, dim1);
            if (g_grad_enabled && self->requires_grad) {
                result->requires_grad = true;
                auto node = std::make_shared<Node>();
                node->inputs = {self};
                node->op_name = "transpose";
                node->backward_fn = [dim0, dim1](std::shared_ptr<Tensor> grad_out) {
                    return std::vector<std::shared_ptr<Tensor>>{grad_out->transpose(dim0, dim1)->contiguous()};
                };
                result->grad_fn = node;
            }
            return result;
        })
        .def("transpose2d", [](std::shared_ptr<Tensor> self) {
            return self->transpose(0, 1);  // kept for backward compat; no grad tracking here, use .T or .transpose()
        })
        .def_property_readonly("T", [](std::shared_ptr<Tensor> self) {
            auto result = self->transpose_all();
            if (g_grad_enabled && self->requires_grad) {
                result->requires_grad = true;
                auto node = std::make_shared<Node>();
                node->inputs = {self};
                node->op_name = "transpose_all";
                node->backward_fn = [](std::shared_ptr<Tensor> grad_out) {
                    return std::vector<std::shared_ptr<Tensor>>{grad_out->transpose_all()->contiguous()};
                };
                result->grad_fn = node;
            }
            return result;
        })
        .def("backward", &tensor_backward, py::arg("grad_output") = nullptr, py::arg("retain_graph") = false)
        .def("zero_grad", &Tensor::zero_grad)
        .def("item", &Tensor::item)
        .def("__repr__", &Tensor::repr)
        .def("__str__", &Tensor::repr)
        .def("__len__", [](Tensor &t) { return t.shape.empty() ? 0 : t.shape[0]; })
        .def("exp", [](std::shared_ptr<Tensor> self) { return dispatch_exp(self); })
        .def("log", [](std::shared_ptr<Tensor> self) { return dispatch_log(self); })
        .def("huber_loss", [](std::shared_ptr<Tensor> self, std::shared_ptr<Tensor> target, float delta) {
            return dispatch_huber_loss(self, target, delta);
        }, py::arg("target"), py::arg("delta") = 1.0f)
        .def("view", [](std::shared_ptr<Tensor> self, std::vector<int> new_shape) {
            auto result = self->view(new_shape);
            if (g_grad_enabled && self->requires_grad) {
                result->requires_grad = true;
                auto node = std::make_shared<Node>();
                node->inputs = {self};
                node->op_name = "view";
                auto orig_shape = self->shape;
                node->backward_fn = [orig_shape](std::shared_ptr<Tensor> grad_out) {
                    return std::vector<std::shared_ptr<Tensor>>{grad_out->reshape(orig_shape)};
                };
                result->grad_fn = node;
            }
            return result;
        })
        .def("reshape", [](std::shared_ptr<Tensor> self, std::vector<int> new_shape) {
            auto result = self->reshape(new_shape);
            if (g_grad_enabled && self->requires_grad) {
                result->requires_grad = true;
                auto node = std::make_shared<Node>();
                node->inputs = {self};
                node->op_name = "reshape";
                auto orig_shape = self->shape;
                node->backward_fn = [orig_shape](std::shared_ptr<Tensor> grad_out) {
                    return std::vector<std::shared_ptr<Tensor>>{grad_out->reshape(orig_shape)};
                };
                result->grad_fn = node;
            }
            return result;
        })
        .def("sum", [](std::shared_ptr<Tensor> self, py::object dim, bool keepdim) {
            if (dim.is_none()) return dispatch_sum_all(self);
            return dispatch_sum_axis(self, dim.cast<int>(), keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
        .def("detach", &Tensor::detach)
        .def("__getitem__", [](std::shared_ptr<Tensor> self, py::object key) -> py::object {
    std::vector<py::object> items;
    if (py::isinstance<py::tuple>(key)) {
        for (auto item : key) items.push_back(py::reinterpret_borrow<py::object>(item));
    } else {
        items.push_back(key);
    }
    size_t ndim = self->shape.size();
    if (items.size() > ndim)
        throw std::out_of_range("Too many indices for tensor of dimension " + std::to_string(ndim));

    std::vector<int> new_shape, new_strides;
    int offset = 0;

    // One IndexSpec per ORIGINAL dimension. Captured by the backward closure
    // so it can reconstruct exactly how each original position maps to (or
    // is dropped from) the sliced/indexed output — the inverse of the
    // forward mapping computed below.
    struct IndexSpec {
        bool is_int;
        int int_index;  // valid when is_int
        int start;      // valid when !is_int
        int step;       // valid when !is_int
    };
    std::vector<IndexSpec> specs(ndim);

    for (size_t d = 0; d < ndim; ++d) {
        if (d < items.size()) {
            py::object sel = items[d];
            if (py::isinstance<py::int_>(sel)) {
                int i = sel.cast<int>();
                int i_orig = i;
                if (i < 0) i += self->shape[d];
                if (i < 0 || i >= self->shape[d])
                    throw std::out_of_range("Index " + std::to_string(i_orig) +
                                             " out of range on dimension " + std::to_string(d) +
                                             " (size " + std::to_string(self->shape[d]) + ")");
                offset += i * self->strides[d];
                specs[d] = {true, i, 0, 0};
            } else if (py::isinstance<py::slice>(sel)) {
                py::slice s = sel.cast<py::slice>();
                size_t start, stop, step, slicelength;
                if (!s.compute((size_t)self->shape[d], &start, &stop, &step, &slicelength))
                    throw std::runtime_error("Invalid slice on dimension " + std::to_string(d));
                offset += (int)start * self->strides[d];
                new_shape.push_back((int)slicelength);
                new_strides.push_back(self->strides[d] * (int)step);
                specs[d] = {false, 0, (int)start, (int)step};
            } else {
                throw std::runtime_error("Index must be int or slice, got " +
                                          py::str(sel.get_type()).cast<std::string>());
            }
        } else {
            // Dimension not indexed at all: implicit full slice, kept as-is.
            new_shape.push_back(self->shape[d]);
            new_strides.push_back(self->strides[d]);
            specs[d] = {false, 0, 0, 1};
        }
    }

    // Zero-copy view in all cases (including full-int indexing, which now
    // yields a 0-d Tensor rather than a raw float — see note above).
    auto result = std::make_shared<Tensor>(self, offset, new_shape, new_strides);

    if (g_grad_enabled && self->requires_grad) {
        result->requires_grad = true;
        auto node = std::make_shared<Node>();
        node->inputs = {self};
        node->op_name = "getitem";
        auto orig_shape = self->shape;
        auto orig_device = self->device;
        auto self_dtype = self->dtype;
        node->backward_fn = [orig_shape, orig_device, self_dtype, specs, new_shape](std::shared_ptr<Tensor> grad_out) {
            auto grad_input = std::make_shared<Tensor>(orig_shape, orig_device, self_dtype);
            grad_input->fill_zero_typed(self_dtype);

            size_t ndim_orig = orig_shape.size();
            size_t ndim_new = new_shape.size();

            int total_new = 1;
            for (int s : new_shape) total_new *= s;

            auto scatter = [&](auto tag) {
                using T = decltype(tag);
                std::vector<int> new_idx(ndim_new, 0);
                for (int flat = 0; flat < total_new; ++flat) {
                    T val = grad_out->get_scalar_typed<T>(new_idx);

                    std::vector<int> orig_idx(ndim_orig);
                    size_t j = 0;
                    for (size_t d = 0; d < ndim_orig; ++d) {
                        if (specs[d].is_int) {
                            orig_idx[d] = specs[d].int_index;
                        } else {
                            orig_idx[d] = specs[d].start + new_idx[j] * specs[d].step;
                            ++j;
                        }
                    }
                    grad_input->set_scalar_typed<T>(orig_idx, val);

                    for (int d = (int)ndim_new - 1; d >= 0; --d) {
                        if (++new_idx[d] < new_shape[d]) break;
                        new_idx[d] = 0;
                    }
                }
            };
            switch (self_dtype) {
                case DType::FLOAT32: scatter(float{});   break;
                case DType::FLOAT64: scatter(double{});  break;
                case DType::INT32:   scatter(int32_t{}); break;
                case DType::INT64:   scatter(int64_t{}); break;
            }

            return std::vector<std::shared_ptr<Tensor>>{grad_input};
        };
        result->grad_fn = node;
    }

    return py::cast(result);
})      
        .def("__add__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return dispatch_add(a, b); })
        .def("__add__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_add_scalar(a, s); })
        .def("__radd__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_add_scalar(a, s); })
        .def("__sub__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return dispatch_sub(a, b); })
        .def("__sub__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_sub_scalar(a, s); })
        .def("__rsub__", [](std::shared_ptr<Tensor> a, float s) {
            auto neg_a = dispatch_mul_scalar(a, -1.0f);
            return dispatch_add_scalar(neg_a, s);
        })
        .def("__mul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return dispatch_mul(a, b); })
        .def("__mul__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_mul_scalar(a, s); })
        .def("__rmul__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_mul_scalar(a, s); })
        .def("__truediv__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return dispatch_div(a, b); })
        .def("__truediv__", [](std::shared_ptr<Tensor> a, float s) { return dispatch_div_scalar(a, s); })
        .def("__rtruediv__", [](std::shared_ptr<Tensor> a, float s) {
            auto s_tensor = dispatch_add_scalar(dispatch_mul_scalar(a, 0.0f), s);
            return dispatch_div(s_tensor, a);
        })
        .def("__matmul__", [](std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) { return dispatch_matmul(a, b); })
        .def("sqrt", [](std::shared_ptr<Tensor> self) { return dispatch_sqrt(self); })
        .def("sign", [](std::shared_ptr<Tensor> self) { return dispatch_sign(self); })
        .def("abs", [](std::shared_ptr<Tensor> self) { return dispatch_abs(self); });

    m.def("_set_grad_enabled", [](bool enabled) { g_grad_enabled = enabled; });
    m.def("_is_grad_enabled", []() { return g_grad_enabled; });

    m.def("fill_cpu_random", &fill_cpu_random, "Fill CPU Tensor with random numbers");
    m.def("fill_cpu_randint", &fill_cpu_randint, "Fill CPU Tensor with random integers");
    m.def("cpu_matmul", &dispatch_matmul, "CPU matrix multiplication (autograd-aware)");
    m.def("cpu_add", &dispatch_add);
    m.def("cpu_sub", &dispatch_sub);
    m.def("from_numpy", &tensor_from_numpy_typed,
      py::arg("array"), py::arg("device") = "cpu", py::arg("requires_grad") = false,
      "Create a Tensor from an existing numpy array, preserving its dtype (float32/float64/int32/int64).");
    m.def("is_available", &cuda_is_available, "Check if a CUDA-capable GPU is actually present and usable");
    m.def("device_count", &cuda_device_count, "Number of CUDA-capable GPUs detected");
    m.def("im2col_1d", &dispatch_im2col_1d,
          py::arg("x"), py::arg("kernel_size"), py::arg("stride"),
          py::arg("padding"), py::arg("dilation"), py::arg("out_length"));
    m.def("col2im_1d", &dispatch_col2im_1d,
      py::arg("col"), py::arg("C"), py::arg("L_in"), py::arg("stride"),
      py::arg("padding"), py::arg("dilation"), py::arg("L_out"));
    m.def("col2im_2d", &dispatch_col2im_2d,
      py::arg("col"), py::arg("C"), py::arg("H_in"), py::arg("W_in"),
      py::arg("SH"), py::arg("SW"), py::arg("PH"), py::arg("PW"), py::arg("DH"), py::arg("DW"),
      py::arg("OH"), py::arg("OW"));
    m.def("col2im_3d", &dispatch_col2im_3d,
      py::arg("col"), py::arg("C"), py::arg("D_in"), py::arg("H_in"), py::arg("W_in"),
      py::arg("SD"), py::arg("SH"), py::arg("SW"), py::arg("PD"), py::arg("PH"), py::arg("PW"),
      py::arg("DD"), py::arg("DH"), py::arg("DW"), py::arg("OD"), py::arg("OH"), py::arg("OW"));
    m.def("_huber_loss_fused", &dispatch_huber_loss, py::arg("pred"), py::arg("target"), py::arg("delta") = 1.0f);
    m.def("_cross_entropy_fused", &dispatch_cross_entropy_fused, py::arg("logits"), py::arg("onehot"));
    m.def("_bce_fused", &dispatch_bce_fused, py::arg("pred"), py::arg("target"));
    m.def("_smoothl1_fused", &dispatch_smoothl1_fused, py::arg("pred"), py::arg("target"), py::arg("beta") = 1.0f);
    m.def("_multimargin_fused", &dispatch_multimargin_fused, py::arg("logits"), py::arg("onehot"), py::arg("margin") = 1.0f);
    m.def("im2col_2d", &dispatch_im2col_2d,
          py::arg("x"), py::arg("KH"), py::arg("KW"), py::arg("SH"), py::arg("SW"),
          py::arg("PH"), py::arg("PW"), py::arg("DH"), py::arg("DW"), py::arg("OH"), py::arg("OW"));
    m.def("im2col_3d", &dispatch_im2col_3d,
      py::arg("x"), py::arg("KD"), py::arg("KH"), py::arg("KW"),
      py::arg("SD"), py::arg("SH"), py::arg("SW"),
      py::arg("PD"), py::arg("PH"), py::arg("PW"),
      py::arg("DD"), py::arg("DH"), py::arg("DW"),
      py::arg("OD"), py::arg("OH"), py::arg("OW"));
    m.def("_adam_step_fused", &dispatch_adam_step_fused,
      py::arg("p"), py::arg("grad"), py::arg("m"), py::arg("v"),
      py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"),
      py::arg("weight_decay"), py::arg("t"));
    m.def("_sgd_step_fused", &dispatch_sgd_step_fused);
    m.def("_adamw_step_fused", &dispatch_adamw_step_fused);
    m.def("_adamax_step_fused", &dispatch_adamax_step_fused);
    m.def("_nadam_step_fused", &dispatch_nadam_step_fused);
    m.def("_radam_step_fused", &dispatch_radam_step_fused);
    m.def("_rmsprop_step_fused", &dispatch_rmsprop_step_fused);
    m.def("_adadelta_step_fused", &dispatch_adadelta_step_fused);
    m.def("_rprop_step_fused", &dispatch_rprop_step_fused);

#ifndef AAKAAR_NO_CUDA
    m.def("generate_random", &run_curand_uniform, "Fill GPU Tensor with random numbers");
    m.def("generate_random", &run_curand_uniform, "Fill GPU Tensor with random numbers");
    m.def("generate_randint", &run_curand_randint);
    m.def("cuda_matmul", &dispatch_matmul, "cuBLAS GPU matrix multiplication");
    m.def("empty_cache", &empty_cache, "Release cached GPU memory");
    m.def("_set_tf32_enabled", &set_tf32_enabled);
    m.def("_get_tf32_enabled", &get_tf32_enabled);
    m.def("_synchronize", &cuda_synchronize);
    m.def("_huber_loss_fused", &dispatch_huber_loss, py::arg("pred"), py::arg("target"), py::arg("delta") = 1.0f);
    m.def("_cross_entropy_fused", &dispatch_cross_entropy_fused, py::arg("logits"), py::arg("onehot"));
    m.def("_mse_fused", &dispatch_mse_fused, py::arg("pred"), py::arg("target"));
    m.def("_nll_fused", &dispatch_nll_fused, py::arg("log_probs"), py::arg("onehot"));
    m.def("_bce_logits_fused", &dispatch_bce_logits_fused, py::arg("logits"), py::arg("target"));
    py::class_<GraphHandle, std::shared_ptr<GraphHandle>>(m, "CudaGraphHandle");
    m.def("_cuda_graph_begin_capture", &cuda_graph_begin_capture);
    m.def("_cuda_graph_end_capture", &cuda_graph_end_capture);
    m.def("_cuda_graph_replay", &cuda_graph_replay);
    m.def("_cuda_graph_synchronize", &cuda_graph_synchronize);
    m.def("_allocator_stats", []() -> py::tuple {
        auto [hits, misses] = CachingAllocator::get_instance().get_stats();
        return py::make_tuple(hits, misses);
    });
    m.attr("HAS_CUDA") = true;

#ifdef AAKAAR_HAS_CUDNN
    m.def("conv1d_cudnn", &dispatch_conv1d_cudnn,
          py::arg("x"), py::arg("w"), py::arg("stride"), py::arg("padding"), py::arg("dilation"));
    m.def("conv1d_transpose_cudnn", &dispatch_conv1d_transpose_cudnn,
      py::arg("x"), py::arg("w"), py::arg("stride"), py::arg("padding"), py::arg("dilation"), py::arg("L_out"));
    m.def("_set_cudnn_tf32_enabled", &set_cudnn_tf32_enabled);
    m.def("_get_cudnn_tf32_enabled", &get_cudnn_tf32_enabled);
    m.def("_conv1d_forward_into", &run_cudnn_conv1d_forward_into);
    m.def("_conv1d_backward_data_into", &run_cudnn_conv1d_backward_data_into);
    m.def("_conv1d_backward_filter_into", &run_cudnn_conv1d_backward_filter_into);
    m.def("_cudnn_set_stream_for_capture", &cudnn_set_stream_for_capture);
    m.def("_cudnn_reset_stream", &cudnn_reset_stream);
    m.def("_cuda_graph_replay_two", &cuda_graph_replay_two);
    m.def("_cuda_graph_replay_full_step", &cuda_graph_replay_full_step);
    m.def("conv2d_cudnn", &dispatch_conv2d_cudnn,
          py::arg("x"), py::arg("w"), py::arg("SH"), py::arg("SW"),
          py::arg("PH"), py::arg("PW"), py::arg("DH"), py::arg("DW"));
    m.def("_conv2d_forward_into", &run_cudnn_conv2d_forward_into);
    m.def("_conv2d_backward_data_into", &run_cudnn_conv2d_backward_data_into);
    m.def("_conv2d_backward_filter_into", &run_cudnn_conv2d_backward_filter_into);
    m.def("conv3d_cudnn", &dispatch_conv3d_cudnn,
      py::arg("x"), py::arg("w"), py::arg("SD"), py::arg("SH"), py::arg("SW"),
      py::arg("PD"), py::arg("PH"), py::arg("PW"), py::arg("DD"), py::arg("DH"), py::arg("DW"));
    m.def("conv2d_transpose_cudnn", &dispatch_conv2d_transpose_cudnn,
      py::arg("x"), py::arg("w"), py::arg("SH"), py::arg("SW"), py::arg("PH"), py::arg("PW"),
      py::arg("DH"), py::arg("DW"), py::arg("OH"), py::arg("OW"));
    m.def("conv3d_transpose_cudnn", &dispatch_conv3d_transpose_cudnn,
      py::arg("x"), py::arg("w"), py::arg("SD"), py::arg("SH"), py::arg("SW"),
      py::arg("PD"), py::arg("PH"), py::arg("PW"), py::arg("DD"), py::arg("DH"), py::arg("DW"),
      py::arg("OD"), py::arg("OH"), py::arg("OW"));
    m.def("_warm_cudnn", &warm_cudnn);
    m.attr("HAS_CUDNN") = true;
#else
    m.attr("HAS_CUDNN") = false;
#endif

#else
    m.attr("HAS_CUDA") = false;
    m.attr("HAS_CUDNN") = false;
#endif
}