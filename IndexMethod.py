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
            
            # Eliminar duplicados (puede ocurrir por condiciones periódicas)
            neighbor_cells = list(set(neighbor_cells))
            
            # Verificar todas las partículas en celdas vecinas
            for other_cell in neighbor_cells:
                for j in self.cells.get(other_cell, []):
                    for i in particles:
                        if i < j:  # Evitar calcular dos veces (dij = dji)
                            # Distancia con condiciones periódicas
                            dx = self.positions[i,0] - self.positions[j,0]
                            dy = self.positions[i,1] - self.positions[j,1]
                            
                            # Aplicar condiciones periódicas
                            dx = dx - round(dx / self.L) * self.L
                            dy = dy - round(dy / self.L) * self.L
                            
                            distance = np.sqrt(dx**2 + dy**2)
                            min_distance = distance - self.radii[i] - self.radii[j]
                            
                            if min_distance < self.rc:
                                self.neighbors[i].append(j)
                                self.neighbors[j].append(i)
    
    def brute_force_neighbors(self):
        """Encuentra vecinos por fuerza bruta (para comparación)"""
        neighbors = { i: [] for i in range(self.N) }
        
        for i in range(self.N):
            for j in range(i+1, self.N):
                # Distancia con condiciones periódicas
                dx = self.positions[i,0] - self.positions[j,0]
                dy = self.positions[i,1] - self.positions[j,1]
                
                # Aplicar condiciones periódicas
                dx = dx - round(dx / self.L) * self.L
                dy = dy - round(dy / self.L) * self.L
                
                distance = np.sqrt(dx**2 + dy**2)
                min_distance = distance - self.radii[i] - self.radii[j]
                
                if min_distance < self.rc:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        
        return neighbors
    
    def update_positions(self, dt=0.1):
        """Actualiza posiciones de las partículas"""
        self.positions += self.velocities * dt
        
        # Aplicar condiciones periódicas de contorno
        self.positions = self.positions % self.L
    
    def save_static_info(self, filename="static_info.txt"):
        """Guarda información estática en archivo"""
        with open(filename, 'w') as f:
            f.write(f"{self.N}\n")
            f.write(f"{self.L}\n")
            for i in range(self.N):
                f.write(f"{self.radii[i]} {int(self.colors[i,0]*255)} {int(self.colors[i,1]*255)} {int(self.colors[i,2]*255)}\n")
    
    def save_dynamic_info(self, filename="dynamic_info.txt", steps=10, dt=0.1):
        """Guarda información dinámica en archivo"""
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
        """Visualiza el sistema"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Dibujar celdas
        for i in range(self.M+1):
            ax.axhline(i * self.cell_size, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(i * self.cell_size, color='gray', linestyle='--', alpha=0.5)
        
        # Dibujar partículas
        for i in range(self.N):
            circle = patches.Circle(self.positions[i], self.radii[i], 
                                    color=self.colors[i], alpha=0.7)
            ax.add_patch(circle)
            ax.text(*self.positions[i], str(i), ha='center', va='center', fontsize=8)
        
        # Dibujar conexiones entre vecinos
        if show_neighbors:
            for i in range(self.N):
                for j in self.neighbors[i]:
                    if i < j:  # Evitar dibujar dos veces
                        # Ajustar para condiciones periódicas
                        dx = self.positions[j,0] - self.positions[i,0]
                        dy = self.positions[j,1] - self.positions[i,1]
                        
                        # Aplicar condiciones periódicas para visualización
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
        
        # Inicializar elementos
        particles = []
        neighbor_lines = []  # Almacenará las líneas entre vecinos
        
        for i in range(self.N):
            circle = patches.Circle(self.positions[i], self.radii[i],
                                color=self.colors[i], alpha=0.7)
            ax.add_patch(circle)
            particles.append(circle)
        
        # Dibujar celdas
        for i in range(self.M+1):
            ax.axhline(i * self.cell_size, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(i * self.cell_size, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_aspect('equal')
        ax.set_title(f"Simulación CIM (N={self.N}, rc={self.rc})")

        def update(frame):
            # Actualizar posiciones
            self.update_positions()
            self.assign_to_cells()
            self.find_neighbors()
            
            # Actualizar partículas
            for i in range(self.N):
                particles[i].center = self.positions[i]
            
            # Limpiar líneas anteriores
            for line in neighbor_lines:
                line.remove()
            neighbor_lines.clear()
            
            # Dibujar nuevas conexiones entre vecinos
            for i in range(self.N):
                for j in self.neighbors[i]:
                    if i < j:  # Evitar duplicados
                        # Ajustar para condiciones periódicas
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
        
        # Crear animación
        ani = FuncAnimation(
            fig, update, frames=frames,
            interval=interval, blit=True
        )
        
        # Guardar GIF
        print(f"Guardando animación como {filename}...")
        ani.save(filename, writer='pillow', fps=1000/interval, dpi=100)
        plt.close(fig)
        
        # Mostrar en Jupyter si es posible
        try:
            from IPython.display import Image
            return Image(filename=filename)
        except:
            print(f"Animación guardada como {filename}")
            return None

def compare_methods(L=20, rc=1.5, N_range=range(50, 1001, 50), M=10):
    """Compara el tiempo de ejecución de CIM vs fuerza bruta"""
    cim_times = []
    brute_times = []
    
    for N in N_range:
        print(f"Probando con N = {N}...")
        sim = CellIndexMethod(L, rc, N, M)
        
        # Medir tiempo CIM
        start = time.time()
        sim.assign_to_cells()
        sim.find_neighbors()
        cim_time = time.time() - start
        cim_times.append(cim_time)
        
        # Medir tiempo fuerza bruta
        start = time.time()
        sim.brute_force_neighbors()
        brute_time = time.time() - start
        brute_times.append(brute_time)
    
    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(N_range, brute_times, 'r-', label='Fuerza Bruta (O(N²))')
    plt.plot(N_range, cim_times, 'b-', label='Cell Index Method (O(N))')
    plt.xlabel('Número de partículas (N)')
    plt.ylabel('Tiempo de ejecución (s)')
    plt.title('Comparación de métodos de detección de vecinos')
    plt.legend()
    plt.grid(True)
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Parámetros de la simulación
    L = 20.0  # Tamaño del sistema
    rc = 2.0  # Radio de corte para vecinos
    N = 100   # Número de partículas
    M = 5     # Número de celdas por lado (MxM celdas totales)
    
    # Crear simulación
    sim = CellIndexMethod(L, rc, N, M)
    
    # Asignar partículas a celdas y encontrar vecinos
    sim.assign_to_cells()
    sim.find_neighbors()
    
    # Guardar información
    sim.save_static_info("static_data.txt")
    sim.save_dynamic_info("dynamic_data.txt", steps=20)
    
    # Visualizar
    sim.visualize()
    
    # Comparar métodos
    compare_methods()
    
    # Opcional: Crear animación (descomentar para ejecutar)
    ani = sim.animate_simulation(frames=300, interval=50)