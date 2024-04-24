import glfw
from OpenGL.GL import *
import numpy as np
from fluid import Fluid  # Assuming your fluid simulation class is defined here

def initialize_window(width, height, title):
    if not glfw.init():
        raise Exception("GLFW can't be initialized")

    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window can't be created")

    glfw.make_context_current(window)
    return window

def render(window, fluid, width, height):
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, 0, height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Draw fluid data
        cell_size = width // fluid.numX
        for i in range(1, fluid.numX - 1):
            for j in range(1, fluid.numY - 1):
                x = i * cell_size
                y = j * cell_size
                value = fluid.m[i * fluid.numY + j]
                glColor3f(value, value, value)  # Assuming the value is normalized
                glBegin(GL_QUADS)
                glVertex2f(x, y)
                glVertex2f(x + cell_size, y)
                glVertex2f(x + cell_size, y + cell_size)
                glVertex2f(x, y + cell_size)
                glEnd()

        glfw.swap_buffers(window)
        update_fluid(fluid)

    glfw.terminate()

def update_fluid(fluid):
    dt = 0.1
    fluid.integrate(dt, gravity=9.8)
    fluid.solve_incompressibility(num_iters=10, dt=dt)
    fluid.advect_vel(dt)
    fluid.advect_smoke(dt)

def main():
    width, height = 800, 600
    window = initialize_window(width, height, "Fluid Simulation with GLFW")
    fluid = Fluid(density=1.0, numX=40, numY=40, h=10)
    render(window, fluid, width, height)

if __name__ == "__main__":
    main()
