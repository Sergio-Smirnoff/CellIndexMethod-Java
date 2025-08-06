import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from itertools import product
import time
import datetime
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

    def find_neighbors(self, periodic=False, output_file=None):
        self.neighbors = {i: [] for i in range(self.N)}
        
        self.cells = {}
        for i in range(self.M):
            for j in range(self.M):
                self.cells[(i, j)] = []
        
        for idx in range(self.N):
            cell_x = int(self.positions[idx, 0] / self.cell_size) % self.M
            cell_y = int(self.positions[idx, 1] / self.cell_size) % self.M
            self.cells[(cell_x, cell_y)].append(idx)

        for (cell_x, cell_y), particles in self.cells.items():
            for dx, dy in [(-1,-1), (-1,0), (-1,1),
                        (0,-1), (0,0), (0,1),
                        (1,-1), (1,0), (1,1)]:
                
                nx, ny = cell_x + dx, cell_y + dy
                if not (0 <= nx < self.M and 0 <= ny < self.M):
                    continue
                
                neighbor_particles = self.cells.get((nx, ny), [])
                
                for i in particles:
                    for j in neighbor_particles:
                        if i >= j:
                            continue
                        
                        dx_pos = self.positions[j, 0] - self.positions[i, 0]
                        dy_pos = self.positions[j, 1] - self.positions[i, 1]                        
                        dx_pos -= round(dx_pos / self.L) * self.L
                        dy_pos -= round(dy_pos / self.L) * self.L

                        distance = np.sqrt(dx_pos**2 + dy_pos**2)
                        edge_distance = distance - (self.radii[i] + self.radii[j])
                        
                        if edge_distance < self.rc:
                            self.neighbors[i].append(j)
                            self.neighbors[j].append(i)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write("Particle Neighbors List\n")
                f.write(f"System Parameters: M={self.M}, rc={self.rc}, L={self.L}\n")
                f.write(f"Periodic Boundaries: {periodic}\n")
                f.write("\nParticle ID: Neighbors\n")
                
                for particle_id in sorted(self.neighbors.keys()):
                    neighbors = sorted(self.neighbors[particle_id])
                    f.write(f"{particle_id}: {neighbors}\n")
            print(f"Neighbor list saved to {output_file}")
            
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
    
    def visualize(self, show_neighbors=True, show_indices=True, show_grid=True, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if show_grid and hasattr(self, 'M'):
            for i in range(self.M + 1):
                ax.axhline(i * self.cell_size, color='gray', linestyle='--', alpha=0.3)
                ax.axvline(i * self.cell_size, color='gray', linestyle='--', alpha=0.3)
        
        for i in range(self.N):
            circle = patches.Circle(
                self.positions[i], 
                self.radii[i],
                color=self.colors[i] if hasattr(self, 'colors') else 'blue',
                alpha=0.7
            )
            ax.add_patch(circle)
            
            if show_indices:
                ax.text(*self.positions[i], str(i), 
                    ha='center', va='center', 
                    fontsize=8, color='black')
        
        if show_neighbors and hasattr(self, 'neighbors'):
            for i in range(self.N):
                for j in self.neighbors[i]:
                    if i < j:
                        ax.plot(
                            [self.positions[i, 0], self.positions[j, 0]],
                            [self.positions[i, 1], self.positions[j, 1]],
                            'r-', alpha=0.3, linewidth=1
                        )
        
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_aspect('equal')
        ax.set_title(f"Cell Index Method (M={self.M}, rc={self.rc})")
        
        plt.savefig("fig1.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def animate_simulation(self, frames=50, interval=100, filename="simulation.gif"):
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
                              filename="simulation_cim.gif"):
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
        ax.set_title(f"Simulation with CIM (N={self.N}, L={self.L}, rc={getattr(self, 'rc', '?')})")

        def update(frame_num):
            frame_data = frames_data[frame_num % len(frames_data)]
            
            for i, (x, y, _, _) in enumerate(frame_data):
                particles[i].center = (x, y)
            
            for line in neighbor_lines:
                line.remove()
            neighbor_lines.clear()
            
            if not hasattr(self, 'M') or not hasattr(self, 'rc'):
                raise ValueError("Cell Index Method not initialized (missing M or rc)")
            
            cells = {}
            for i in range(self.M):
                for j in range(self.M):
                    cells[(i, j)] = []
            
            for idx, (x, y, _, _) in enumerate(frame_data):
                cell_x = int(x / self.cell_size) % self.M
                cell_y = int(y / self.cell_size) % self.M
                cells[(cell_x, cell_y)].append(idx)
            
            for (cell_x, cell_y), particles_in_cell in cells.items():
                for dx, dy in [(-1,-1), (-1,0), (-1,1),
                            (0,-1), (0,0), (0,1),
                            (1,-1), (1,0), (1,1)]:
                    nx, ny = (cell_x + dx) % self.M, (cell_y + dy) % self.M
                    
                    for i in particles_in_cell:
                        for j in cells.get((nx, ny), []):
                            if i < j:
                                xi, yi, _, _ = frame_data[i]
                                xj, yj, _, _ = frame_data[j]
                                
                                dx_pos = xj - xi
                                dy_pos = yj - yi
                                dx_pos -= round(dx_pos / self.L) * self.L
                                dy_pos -= round(dy_pos / self.L) * self.L
                                
                                distance = (dx_pos**2 + dy_pos**2)**0.5
                                edge_distance = distance - (self.radii[i] + self.radii[j])
                                
                                if edge_distance < self.rc:
                                    line, = ax.plot([xi, xi+dx_pos], [yi, yi+dy_pos],
                                                'r-', alpha=0.3, lw=1)
                                    neighbor_lines.append(line)
            
            return particles + neighbor_lines
        
        ani = FuncAnimation(fig, update, frames=min(frames, len(frames_data)), 
                        interval=interval, blit=True)
        
        print(f"Saving CIM animation to {filename}...")
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

    file = open("./times/tiemposyN.txt", "w")
    file.write("N\tTiempo (s)\n")
    
    for N in N_range:
        print(f"Probando con N = {N}...")
        sim = CellIndexMethod(L, rc, N, M)
        
        start = time.time()
        sim.assign_to_cells()
        sim.find_neighbors()
        cim_time = time.time() - start
        cim_times.append(cim_time)

        file.write(f"{N}\t{cim_time:.4f}\n")
        
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

def compare_methodsCells(L=20, rc=1.5, M_range=range(1, 20), N=100):
    cim_times = []
    brute_times = []

    file = open(f"./times/tiemposyM{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w")
    file.write("M\tTiempo (s)\n")
    
    for M in M_range:
        print(f"Probando con M = {M}...")
        sim = CellIndexMethod(L, rc, N, M)
        
        start = time.time()
        sim.assign_to_cells()
        sim.find_neighbors()
        cim_time = time.time() - start
        cim_times.append(cim_time)
        file.write(f"{M}\t{cim_time:.4f}\n")

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