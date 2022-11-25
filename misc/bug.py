import taichi as ti

ti.init(arch=ti.gpu)

bar = ti.Matrix.field(3, 3, ti.float32, shape=16)
zoo = ti.Vector.field(3, ti.float32, shape=16)

bar.fill(0)
zoo.fill(1)

bar[0] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@ti.kernel
def foo():

    ti.loop_config(serialize=True)
    for i in range(10):
        print("start ", i)
        res = ti.solve(bar[i], zoo[i], ti.float32)
        print(res)
        print("end ", i)


foo()
