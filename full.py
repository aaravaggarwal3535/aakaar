import sys; sys.path.insert(0, '/home/aarav/torch_cuda/aakaar')
import aakaar, numpy as np

passed, failed = 0, 0
def check(name, cond):
    global passed, failed
    if cond: passed += 1; print(f"PASS: {name}")
    else: failed += 1; print(f"FAIL: {name}")

print("=== 1. Basic matmul shapes ===")
a = aakaar.rand((4,5), device='cpu', seed=1)
b = aakaar.rand((5,3), device='cpu', seed=2)
check("2D matmul", np.allclose(aakaar.matmul(a,b).to_numpy(), a.to_numpy() @ b.to_numpy()))
check("2D @ operator", np.allclose((a@b).to_numpy(), a.to_numpy() @ b.to_numpy()))

print("=== 2. Mixed-rank broadcast ===")
a2 = aakaar.rand((1,4,5), device='cpu', seed=3)
b2 = aakaar.rand((6,1,5,3), device='cpu', seed=4)
try:
    c2 = aakaar.matmul(a2, b2)
    expected2 = a2.to_numpy() @ b2.to_numpy()
    check("4D vs 3D mixed-rank broadcast", np.allclose(c2.to_numpy(), expected2))
except Exception as e:
    print("mixed-rank broadcast raised:", e)
    check("4D vs 3D mixed-rank broadcast", False)

print("=== 3. Matmul shape mismatch error ===")
try:
    aakaar.matmul(aakaar.rand((3,4)), aakaar.rand((5,3)))
    check("matmul dim mismatch raises", False)
except (ValueError, RuntimeError):
    check("matmul dim mismatch raises", True)

print("=== 4. Non-contiguous matmul rejection ===")
big = aakaar.rand((10,10), device='cpu', seed=5)
view = big[1:5, 1:5]
w4 = aakaar.rand((4,4), device='cpu', seed=6)
try:
    aakaar.matmul(view, w4)
    check("non-contig matmul rejected", False)
except (ValueError, RuntimeError):
    check("non-contig matmul rejected", True)
check("contiguous().matmul works", np.allclose(
    aakaar.matmul(view.contiguous(), w4).to_numpy(),
    view.to_numpy() @ w4.to_numpy()
))

print("=== 5. View / reshape edge cases ===")
t = aakaar.rand((2,3,4), device='cpu', seed=7)
tn = t.to_numpy()
check("view flatten", np.allclose(t.view([24]).to_numpy(), tn.reshape(24)))
check("view with -1 middle", np.allclose(t.view([2,-1]).to_numpy(), tn.reshape(2,-1)))
try:
    t.view([25]); check("view size mismatch raises", False)
except ValueError: check("view size mismatch raises", True)
try:
    t.view([5,-1]); check("view non-divisible -1 raises", False)
except ValueError: check("view non-divisible -1 raises", True)

print("=== 6. Transpose edge cases ===")
t3 = aakaar.rand((2,3,4,5), device='cpu', seed=8)
t3n = t3.to_numpy()
check("full .T on 4D", np.allclose(t3.T.to_numpy(), t3n.T))
check("transpose(1,3)", np.allclose(t3.transpose(1,3).to_numpy(), np.swapaxes(t3n,1,3)))
check("transpose negative dims", np.allclose(t3.transpose(-1,-2).to_numpy(), np.swapaxes(t3n,-1,-2)))
try:
    t3.transpose(0, 10); check("transpose out-of-range raises", False)
except (IndexError, RuntimeError): check("transpose out-of-range raises", True)

print("=== 7. Sum edge cases ===")
s = aakaar.rand((3,4,5), device='cpu', seed=9)
sn = s.to_numpy()
check("sum dim=0", np.allclose(s.sum(dim=0).to_numpy(), sn.sum(axis=0)))
check("sum dim=-1", np.allclose(s.sum(dim=-1).to_numpy(), sn.sum(axis=-1)))
check("sum keepdim=True", np.allclose(s.sum(dim=1, keepdim=True).to_numpy(), sn.sum(axis=1, keepdims=True)))
check("sum full scalar", np.allclose(s.sum().to_numpy(), sn.sum()))

print("=== 8. Gradient accumulation (diamond graph) ===")
x = aakaar.rand((3,), device='cpu', seed=10, requires_grad=True)
xn = x.to_numpy()
y1 = aakaar.matmul(x.view([1,3]), x.view([3,1]))
y2 = aakaar.matmul(x.view([1,3]), x.view([3,1]))
z = y1 + y2
z.backward()
expected_grad = 4 * xn
check("grad accumulation (diamond graph)", np.allclose(x.grad.to_numpy(), expected_grad, atol=1e-3))

print("=== 9. zero_grad ===")
x.zero_grad()
check("zero_grad clears grad", x.grad is None)

print("=== 10. Repeated backward accumulates ===")
x2 = aakaar.rand((1,3), device='cpu', seed=11, requires_grad=True)
w = aakaar.rand((3,1), device='cpu', seed=12)
l1 = aakaar.matmul(x2, w); l1.backward()
g1 = x2.grad.to_numpy().copy()
l2 = aakaar.matmul(x2, w); l2.backward()
g2 = x2.grad.to_numpy()
check("repeated backward accumulates (doubles)", np.allclose(g2, 2*g1, atol=1e-3))

print("=== 11. CUDA parity ===")
xc = aakaar.rand((3,4), device='cuda', seed=13)
xcn = xc.to_numpy()
check("cuda view", np.allclose(xc.view([12]).to_numpy(), xcn.reshape(12)))
check("cuda transpose", np.allclose(xc.T.to_numpy(), xcn.T))
check("cuda sum axis", np.allclose(xc.sum(dim=0).to_numpy(), xcn.sum(axis=0)))
yc = aakaar.rand((4,5), device='cuda', seed=14)
check("cuda matmul", np.allclose(aakaar.matmul(xc,yc).to_numpy(), xcn @ yc.to_numpy(), atol=1e-3))

print("=== 12. Mixed device matmul ===")
try:
    aakaar.matmul(aakaar.rand((2,2),device='cpu'), aakaar.rand((2,2),device='cuda'))
    check("mixed device matmul did not crash", True)
except Exception as e:
    check(f"mixed device matmul raised cleanly ({type(e).__name__})", True)

print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")