import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from itertools import product
import time
from IPython.display import HTML

class CellIndexMethod:
    def __init__(self, L, rc, N=100, M=10):
        """
        Inicializa la simulación
        
        Parámetros:
        L: tamaño del sistema (cuadrado LxL)
        rc: radio de corte para vecinos
        N: número de partículas
        M: número de celdas por lado
        """
        self.L = L
        self.rc = rc
        self.N = N
        self.M = M
        self.cell_size = L / M
        
        # Verificar condición de tamaño de celda
        if self.cell_size < rc:
            print(f"Advertencia: Tamaño de celda ({self.cell_size}) menor que rc ({rc}).")
            print("Recomendación: Reducir M para que L/M > rc")
        
        # Inicializar partículas
        self.positions = np.random.rand(N, 2) * L
        self.velocities = (np.random.rand(N, 2) - 0.5) * 2
        self.radii = np.random.uniform(0.1, 0.3, N)
        self.colors = np.random.rand(N, 3)
        
        # Estructuras para el CIM
        self.cells = {}
        self.neighbors = {}
        
    def assign_to_cells(self):
        """Asigna partículas a celdas"""
        self.cells = { (i,j): [] for i in range(self.M) for j in range(self.M) }
        
        for idx in range(self.N):
            x, y = self.positions[idx]
            cell_x = int(x / self.cell_size)
            cell_y = int(y / self.cell_size)
            
            # Aplicar condiciones periódicas de contorno
            cell_x = cell_x % self.M
            cell_y = cell_y % self.M
            
            self.cells[(cell_x, cell_y)].append(idx)
    
    def find_neighbors(self):
        """Encuentra vecinos usando CIM"""
        self.neighbors = { i: [] for i in range(self.N) }
        
        for (cell_x, cell_y), particles in self.cells.items():
            # Obtener celdas vecinas (incluyendo la actual)
            neighbor_cells = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    # Condiciones periódicas de contorno
                    nx = (cell_x + dx) % self.M
                    ny = (cell_y + dy) % self.M
                    neighbor_cells.append((nx, ny))
            

            neighbor_cells = list(set(neighbor_cells))
            
            for other_cell in neighbor_cells:
                for j in self.cells.get(other_cell, []):
                    for i in particles:
                        if i < j:
                            dx = self.positions[i,0] - self.positions[j,0]
                            dy = self.positions[i,1] - self.positions[j,1]
                            
                            dx = dx - round(dx / self.L) * self.L
                            dy = dy - round(dy / self.L) * self.L
                            
                            distance = np.sqrt(dx**2 + dy**2)
                            min_distance = distance - self.radii[i] - self.radii[j]
                            
                            if min_distance < self.rc:
                                self.neighbors[i].append(j)
                                self.neighbors[j].append(i)
    
    def brute_force_neighbors(self):
        neighbors = { i: [] for i in range(self.N) }
        
        for i in range(self.N):
            for j in range(i+1, self.N):
                dx = self.positions[i,0] - self.positions[j,0]
                dy = self.positions[i,1] - self.positions[j,1]
                
                dx = dx - round(dx / self.L) * self.L
                dy = dy - round(dy / self.L) * self.L
                
                distance = np.sqrt(dx**2 + dy**2)
                min_distance = distance - self.radii[i] - self.radii[j]
                
                if min_distance < self.rc:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        
        return neighbors
    
    def update_positions(self, dt=0.1):
        self.positions += self.velocities * dt
        
        self.positions = self.positions % self.L
    
    def save_static_info(self, filename="static_info.txt"):
        with open(filename, 'w') as f:
            f.write(f"{self.N}\n")
            f.write(f"{self.L}\n")
            for i in range(self.N):
                f.write(f"{self.radii[i]} {int(self.colors[i,0]*255)} {int(self.colors[i,1]*255)} {int(self.colors[i,2]*255)}\n")
    
    def save_dynamic_info(self, filename="dynamic_info.txt", steps=10, dt=0.1):
        with open(filename, 'w') as f:
            for step in range(steps):
                f.write(f"t{step}\n")
                for i in range(self.N):
                    x, y = self.positions[i]
                    vx, vy = self.velocities[i]
                    f.write(f"{x} {y} {vx} {vy}\n")
                
                self.update_positions(dt)
                self.assign_to_cells()
                self.find_neighbors()
    
    def visualize(self, show_neighbors=True):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        for i in range(self.M+1):
            ax.axhline(i * self.cell_size, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(i * self.cell_size, color='gray', linestyle='--', alpha=0.5)
        
        for i in range(self.N):
            circle = patches.Circle(self.positions[i], self.radii[i], 
                                    color=self.colors[i], alpha=0.7)
            ax.add_patch(circle)
            ax.text(*self.positions[i], str(i), ha='center', va='center', fontsize=8)
        
        if show_neighbors:
            for i in range(self.N):
                for j in self.neighbors[i]:
                    if i < j:
                        dx = self.positions[j,0] - self.positions[i,0]
                        dy = self.positions[j,1] - self.positions[i,1]
                        
                        dx = dx - round(dx / self.L) * self.L
                        dy = dy - round(dy / self.L) * self.L
                        
                        ax.plot([self.positions[i,0], self.positions[i,0]+dx],
                                [self.positions[i,1], self.positions[i,1]+dy],
                                'r-', alpha=0.3)
        
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_aspect('equal')
        ax.set_title(f"Cell Index Method (M={self.M}, rc={self.rc})")
        plt.show()
    
        
    def animate_simulation(self, frames=50, interval=100, filename="simulation.gif"):
        """Crea y guarda una animación GIF que muestra las conexiones entre vecinos"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        particles = []
        neighbor_lines = []
        
        for i in range(self.N):
            circle = patches.Circle(self.positions[i], self.radii[i],
                                color=self.colors[i], alpha=0.7)
            ax.add_patch(circle)
            particles.append(circle)
        
        for i in range(self.M+1):
            ax.axhline(i * self.cell_size, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(i * self.cell_size, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_aspect('equal')
        ax.set_title(f"Simulación CIM (N={self.N}, rc={self.rc})")

        def update(frame):
            self.update_positions()
            self.assign_to_cells()
            self.find_neighbors()
            
            for i in range(self.N):
                particles[i].center = self.positions[i]
            
            for line in neighbor_lines:
                line.remove()
            neighbor_lines.clear()
            
            for i in range(self.N):
                for j in self.neighbors[i]:
                    if i < j:
                        dx = self.positions[j,0] - self.positions[i,0]
                        dy = self.positions[j,1] - self.positions[i,1]
                        
                        dx = dx - round(dx/self.L)*self.L
                        dy = dy - round(dy/self.L)*self.L
                        
                        line, = ax.plot(
                            [self.positions[i,0], self.positions[i,0]+dx],
                            [self.positions[i,1], self.positions[i,1]+dy],
                            'r-', alpha=0.4, lw=1
                        )
                        neighbor_lines.append(line)
            
            return particles + neighbor_lines
        
        ani = FuncAnimation(
            fig, update, frames=frames,
            interval=interval, blit=True
        )
        
        print(f"Guardando animación como {filename}...")
        ani.save(filename, writer='pillow', fps=1000/interval, dpi=100)
        plt.close(fig)
        
        try:
            from IPython.display import Image
            return Image(filename=filename)
        except:
            print(f"Animación guardada como {filename}")
            return None

    def animate_simulation_from_data(self, frames=50, interval=100, 
                        dynamic_data="dynamic_data.txt", 
                        static_data="static_data.txt", 
                        filename="simulation2.gif"):
        """Creates animation from saved data files including neighbor connections"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.animation import FuncAnimation
        from IPython.display import Image
        
        with open(static_data, 'r') as f:
            self.N = int(f.readline())
            self.L = float(f.readline())
            self.radii = []
            self.colors = []
            for _ in range(self.N):
                parts = f.readline().split()
                self.radii.append(float(parts[0]))
                self.colors.append([float(c)/255 for c in parts[1:4]])
        
        with open(dynamic_data, 'r') as f:
            frames_data = []
            current_frame = []
            for line in f:
                if line.startswith('t'):
                    if current_frame:
                        frames_data.append(current_frame)
                        current_frame = []
                else:
                    parts = list(map(float, line.split()))
                    current_frame.append(parts)
            if current_frame:
                frames_data.append(current_frame)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        particles = []
        neighbor_lines = []
        
        for i in range(self.N):
            circle = patches.Circle((0, 0), self.radii[i], 
                                color=self.colors[i], alpha=0.7)
            ax.add_patch(circle)
            particles.append(circle)
        
        if hasattr(self, 'M'):
            for i in range(self.M+1):
                ax.axhline(i * self.cell_size, color='gray', linestyle='--', alpha=0.3)
                ax.axvline(i * self.cell_size, color='gray', linestyle='--', alpha=0.3)
        
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_aspect('equal')
        ax.set_title(f"Simulation (N={self.N}, L={self.L})")

        def update(frame_num):
            frame_data = frames_data[frame_num % len(frames_data)]
            
            for i, (x, y, vx, vy) in enumerate(frame_data):
                particles[i].center = (x, y)
            
            for line in neighbor_lines:
                line.remove()
            neighbor_lines.clear()
            
            for i in range(self.N):
                for j in range(i+1, self.N):
                    xi, yi, _, _ = frame_data[i]
                    xj, yj, _, _ = frame_data[j]
                    
                    # Periodic boundary conditions
                    dx = xj - xi
                    dy = yj - yi
                    dx = dx - round(dx/self.L)*self.L
                    dy = dy - round(dy/self.L)*self.L
                    distance = (dx**2 + dy**2)**0.5
                    
                    if distance < self.rc + self.radii[i] + self.radii[j]:
                        line, = ax.plot([xi, xi+dx], [yi, yi+dy], 
                                    'r-', alpha=0.3, lw=1)
                        neighbor_lines.append(line)
            
            return particles + neighbor_lines
        
        ani = FuncAnimation(fig, update, frames=min(frames, len(frames_data)), 
                        interval=interval, blit=True)
        
        print(f"Saving animation to {filename}...")
        ani.save(filename, writer='pillow', fps=1000/interval, dpi=100)
        plt.close(fig)
        
        try:
            from IPython.display import Image
            return Image(filename=filename)
        except:
            print(f"Animation saved to {filename}")
        return None 

def compare_methods(L=20, rc=1.5, N_range=range(50, 1001, 50), M=10):
    cim_times = []
    brute_times = []
    
    for N in N_range:
        print(f"Probando con N = {N}...")
        sim = CellIndexMethod(L, rc, N, M)
        
        start = time.time()
        sim.assign_to_cells()
        sim.find_neighbors()
        cim_time = time.time() - start
        cim_times.append(cim_time)
        
        start = time.time()
        sim.brute_force_neighbors()
        brute_time = time.time() - start
        brute_times.append(brute_time)
    
    plt.figure(figsize=(10, 6))
    plt.plot(N_range, brute_times, 'r-', label='Fuerza Bruta (O(N²))')
    plt.plot(N_range, cim_times, 'b-', label='Cell Index Method (O(N))')
    plt.xlabel('Número de partículas (N)')
    plt.ylabel('Tiempo de ejecución (s)')
    plt.title('Comparación de métodos de detección de vecinos')
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_methodsCells(L=20, rc=1.5, M_range=range(1, 15), N=100):
    cim_times = []
    brute_times = []
    
    for M in M_range:
        print(f"Probando con M = {M}...")
        sim = CellIndexMethod(L, rc, N, M)
        
        start = time.time()
        sim.assign_to_cells()
        sim.find_neighbors()
        cim_time = time.time() - start
        cim_times.append(cim_time)
        
        start = time.time()
        sim.brute_force_neighbors()
        brute_time = time.time() - start
        brute_times.append(brute_time)
    
    plt.figure(figsize=(10, 6))
    plt.plot(M_range, brute_times, 'r-', label='Fuerza Bruta (O(N²))')
    plt.plot(M_range, cim_times, 'b-', label='Cell Index Method (O(N))')
    plt.xlabel('Número de celdas (M)')
    plt.ylabel('Tiempo de ejecución (s)')
    plt.title('Comparación de métodos de detección de vecinos')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("comparacion_metodos.png", dpi=150)

if __name__ == "__main__":
    L = 20.0
    rc = 2.0
    N = 100
    M = 5
    
    sim = CellIndexMethod(L, rc, N, M)
    
    sim.assign_to_cells()
    sim.find_neighbors()
    
    sim.save_static_info("static_data.txt")
    sim.save_dynamic_info("dynamic_data.txt", steps=20)
    
    sim.visualize()
    
    compare_methodsCells()
    
    #ani = sim.animate_simulation(frames=300, interval=50)
    ani2 = sim.animate_simulation_from_data()