import taichi as ti

ti.init(arch=ti.cpu)

bar = ti.Matrix.field(3, 3, ti.float32, shape=16)
zoo = ti.Vector.field(3, ti.float32, shape=16)

bar.fill(0)
zoo.fill(1)

bar[0] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@ti.kernel
def foo():
    A = ti.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    ti.loop_config(serialize=True)
    for i in range(10):
        print("start ", i)
        # ti.solve(A, zoo[0], ti.float32)
        res = ti.solve(bar[0], zoo[i], ti.float32)
        print(res)
        print("end ", i)


foo()
