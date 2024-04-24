import numpy as np


class Fluid:
    def __init__(self, density, numX, numY, h) -> None:
        self.density = density 
        self.numX = numX + 2  # Add boundary cells horizontally
        self.numY = numY + 2  # Add boundary cells vertically
        self.numCells = self.numX * self.numY  # Corrected to multiply numX and numY
        self.h = h

        # Initialize arrays for fluid dynamics properties
        self.u = np.zeros(self.numCells, dtype=np.float32)  # Horizontal velocities
        self.v = np.zeros(self.numCells, dtype=np.float32)  # Vertical velocities
        self.newU = np.zeros(self.numCells, dtype=np.float32)
        self.newV = np.zeros(self.numCells, dtype=np.float32)
        self.p = np.zeros(self.numCells, dtype=np.float32)  # Pressure
        self.s = np.zeros(self.numCells, dtype=np.float32)  # Status of each cell (solid/fluid)
        self.m = np.ones(self.numCells, dtype=np.float32)  # Example scalar field (e.g., density)
        self.newM = np.zeros(self.numCells, dtype=np.float32)

        self.initialize_boundaries()

    def initialize_boundaries(self):
        n = self.numY
        # Set the boundary cells' statuses
        for i in range(self.numX):
            self.s[i * n] = 0  # Left boundary for each row
            self.s[i * n + self.numY - 1] = 0  # Right boundary for each row
        
        # Set top and bottom boundaries
        for j in range(self.numY):
            self.s[j] = 0  # Top boundary of the first row
            self.s[(self.numX - 1) * n + j] = 0  # Bottom boundary of the last row

    
    def integrate(self, dt, gravity):
        n = self.numY
        # Apply gravity only to non-boundary fluid cells
        for i in range(1, self.numX - 1):
            for j in range(1, self.numY - 1):
                index = i * n + j
                if self.s[index] != 0:  # Only update if the cell is not a boundary/solid
                    self.v[index] += gravity * dt
    
    def solve_incompressibility(self, num_iters, dt):
        n = self.numY 
        cp = self.density * self.h/dt 

        for _ in range(num_iters):
            for i in range(1, self.numX - 1):
                for j in range(1, self.numY -1):
                    if self.s[i*n + j] == 0:
                        continue
                    s = self.s[i*n+ j]
                    sx0 = self.s[(i-1) * n + j]
                    sx1 = self.s[(i + 1) * n + j]
                    sy0 = self.s[i * n + j - 1]
                    sy1 = self.s[i * n + j + 1]
                    s = sx0 + sx1 + sy0 + sy1
                    if s == 0:
                        continue
                    div = self.u[(i + 1) * n + j] - self.u[i * n + j] + \
                    self.v[i* n + j + 1] - self.v[i * n + j]

                    p = -div/s 
                    p *= 1.0
                    self.p[i * n + j] += cp * p
                    
                    self.u[i * n + j] -= sx0 * p
                    self.u[(i + 1) * n + j] += sx1 * p
                    self.v[i * n + j] -= sy0 * p
                    self.v[i * n + j + 1] += sy1 * p

    def sample_field(self, x, y, field):
        n = self.numY 
        h1 = 1.0 / self.h
        dx,dy = 0.0, 0.0
        if field == 'u':
            f = self.u
            dy = 0.5 * self.h
        elif field == 'v':
            f = self.v 
            dx = 0.5 * self.h
        elif field == 'm':
            f = self.m
            dx = dy = 0.5 * self.h
        x0 = int(min((x - dx) * h1, self.numX - 1))
        x1 = min(x0 + 1, self.numX - 1)
        y0 = int(min((y - dy) * h1, self.numY - 1))
        y1 = min(y0 + 1, self.numY - 1)

        tx = (x - dx) * h1 - x0
        ty = (y - dy) * h1 - y0

        sx = 1.0 - tx
        sy = 1.0 - ty

        return (sx * sy * f[x0 * n + y0] +
                tx * sy * f[x1 * n + y0] +
                tx * ty * f[x1 * n + y1] +
                sx * ty * f[x0 * n + y1])
    
    def advect_vel(self, dt):
        n = self.numY
        self.newU = np.copy(self.u)
        self.newV = np.copy(self.v)

        for i in range(1, self.numX):
            for j in range(1, self.numY):
                if self.s[i * n + j] != 0.0 and self.s[(i - 1) * n + j] != 0.0 and j < self.numY - 1:
                    x, y = i * self.h, j * self.h + 0.5 * self.h
                    u, v = self.u[i*n + j], self.sample_field(x,y, 'v') 
                    x -= dt * u
                    y -= dt * v
                    self.newV[i * n + j] = self.sample_field(x, y, 'v')
        self.u, self.v = self.newU, self.newV
    
    def advect_smoke(self, dt):
        n = self.numY
        self.newM = np.copy(self.m)

        for i in range(1, self.numX - 1):
            for j in range(1, self.numY - 1):
                if self.s[i * n + j] != 0.0:
                    u = (self.u[i * n + j] + self.u[(i + 1) * n + j]) * 0.5
                    v = (self.v[i * n + j] + self.v[i * n + j + 1]) * 0.5
                    x, y = i * self.h + 0.5 * self.h - dt * u, j * self.h + 0.5 * self.h - dt * v
                    self.newM[i * n + j] = self.sample_field(x, y, 'm')

        self.m = np.copy(self.newM)

