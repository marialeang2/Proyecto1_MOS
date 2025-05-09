# %%
import matplotlib.pyplot as plt
import pandas as pd
import folium 
import random
import numpy as np
from pyomo.environ import *

# %%
def load_distance_time_matrix(path, N):
    data= pd.read_csv(path)
    distance = np.zeros((N,N))
    time = np.zeros((N,N))
    
    for i in range(len(data)):
        origen = int(data.iloc[i, 0])
        destino = int(data.iloc[i, 1])
        distance[origen-1, destino-1] = float(data.iloc[i, 2])
        time[origen-1, destino-1] = float(data.iloc[i, 3])
    return distance, time

def load_vehicles(path):
    data= pd.read_csv(path)
    N = len(data)
    vehicles = {}
    for i in range(N):
        id = int(data.iloc[i, 0])
        capacity = int(data.iloc[i, 1])
        ran = float(data.iloc[i, 2])
        vehicles[id] = (capacity, ran)
    
    return vehicles

def load_demand(clientsPath):
    data= pd.read_csv(clientsPath)
    N = len(data)
    demand_dic = {}
    for i in range(N):
        id = int(data.iloc[i, 1])
        demand = float(data.iloc[i, 2])
        demand_dic[id] = demand
    
    return demand_dic


def load_capacity(depositPath):
    data= pd.read_csv(depositPath)
    N = len(data)
    capacity_dic = {}
    for i in range(N):
        id = int(data.iloc[i, 1])
        capacity = float(data.iloc[i, 4])
        capacity_dic[id] = capacity
    
    return capacity_dic

def load_distance_time_dic(path):
    data= pd.read_csv(path)
    distance = {}
    time = {}
    
    for i in range(len(data)):
        origen = int(data.iloc[i, 0])
        destino = int(data.iloc[i, 1])
        distance[origen, destino] = float(data.iloc[i, 2])
        time[origen, destino] = float(data.iloc[i, 3])
    
    return distance, time

def load_coordinates(depotsPath, clientsPath):
    coord = {}
    depot= pd.read_csv(depotsPath)
    client= pd.read_csv(clientsPath)
    
    for i in range(len(depot)):
        id = int(depot.iloc[i, 1])
        lat = float(depot.iloc[i, 3])
        long = float(depot.iloc[i, 2])
        coord[id] = [lat, long]

    for j in range(len(client)):
        id = int(client.iloc[j, 1])
        lat = float(client.iloc[j, 4])
        long = float(client.iloc[j, 3])
        coord[id] = [lat, long]
    
    return coord
        

# %%
#Cargar datos
distancia_dic,_ = load_distance_time_dic('../Datos/Caso_Base/casoBase.csv')
vehiculos = load_vehicles('../Datos/Caso_Base/vehicles.csv')
demanda = load_demand('..\Datos\Caso_Base\clients.csv')
coord = load_coordinates('..\Datos\Caso_Base\depots.csv', '..\Datos\Caso_Base\clients.csv')

np.random.seed(42)
n = 25  # 24 clientes + 1 depósito

dist_matrix_np, time_matrix = load_distance_time_matrix('..\datos\Caso_Base\casoBase.csv', 25)
# Convertir a diccionario {(i,j): distancia}
distancia = {(i, j): dist_matrix_np[i][j]/1000 for i in range(n) for j in range(n)}
print(f"Distancia: {distancia}")
print(f"Vehiculos: {vehiculos}")
print(f"Demanda: {demanda}")


# %%
from pyomo.environ import *

# Crear el modelo
model = ConcreteModel()

# Datos: nodos, demanda, vehículos, distancias
# Suponiendo que ya tienes definidos:
# - demanda: {cliente_id: valor}
# - vehiculos: {vehiculo_id: (capacidad, rango)}
# - distancia: {(i, j): valor}
# - coord: {nodo: (x, y)} para mostrar rutas al final

nodos = [1] + list(demanda.keys())  # Nodo 1 es el depósito
model.N = Set(initialize=nodos)
model.C = Set(initialize=demanda.keys())
model.V = Set(initialize=vehiculos.keys())
model.D = Set(initialize=[1])  # depósito

# Distancias modificadas sin diagonales
offset = 1  # Si distancia usa índices base 0
dist_modificada = {(i + offset, j + offset): v for (i, j), v in distancia.items() if i != j}

# Conjunto de aristas válidas (sin (i, i))
model.A = Set(dimen=2, initialize=dist_modificada.keys())

# Variables
model.x = Var(model.A, model.V, domain=Binary)  # Ruta tomada por vehículo
model.u = Var(model.N, model.V, domain=NonNegativeReals)  # Carga acumulada

# Parámetros
model.dist = Param(model.A, initialize=dist_modificada, within=PositiveReals)
model.demand = Param(model.C, initialize=demanda)
model.cap = Param(model.V, initialize={v: vehiculos[v][0] for v in vehiculos})
model.range = Param(model.V, initialize={v: vehiculos[v][1] for v in vehiculos})

# Función objetivo
def obj_rule(m):
    return sum(m.dist[i, j] * m.x[i, j, v] for (i, j) in m.A for v in m.V)
model.obj = Objective(rule=obj_rule, sense=minimize)

# Restricciones

# Cada cliente es visitado exactamente una vez
model.visit_once = ConstraintList()
for c in model.C:
    model.visit_once.add(
        sum(model.x[i, c, v] for (i, j) in model.A if j == c for v in model.V) == 1
    )

# Flujo conservado
model.flow_conservation = ConstraintList()
for v in model.V:
    for c in model.C:
        model.flow_conservation.add(
            sum(model.x[i, c, v] for (i, j) in model.A if j == c) ==
            sum(model.x[c, j, v] for (i, j) in model.A if i == c)
        )

# Salida y retorno al depósito por vehículo
model.depot_constraints = ConstraintList()
for v in model.V:
    model.depot_constraints.add(
        sum(model.x[1, j, v] for (i, j) in model.A if i == 1) == 1
    )
    model.depot_constraints.add(
        sum(model.x[i, 1, v] for (i, j) in model.A if j == 1) == 1
    )

# Capacidad
model.capacity_constraint = ConstraintList()
for v in model.V:
    model.capacity_constraint.add(
        sum(model.demand[c] * sum(model.x[i, c, v] for (i, j) in model.A if j == c) for c in model.C) <= model.cap[v]
    )

# Carga inicial en el depósito
model.initial_load = ConstraintList()
for v in model.V:
    model.initial_load.add(model.u[1, v] == 0)

# Carga no excede la capacidad
model.max_capacity = ConstraintList()
for v in model.V:
    for i in model.N:
        model.max_capacity.add(model.u[i, v] <= model.cap[v])

# Subtour elimination (MTZ)
model.subtour_elimination = ConstraintList()
for v in model.V:
    for i in model.C:
        for j in model.C:
            if i != j and (i, j) in model.A:
                model.subtour_elimination.add(
                    model.u[i, v] - model.u[j, v] + model.cap[v] * model.x[i, j, v] <= model.cap[v] - model.demand[j]
                )

# Rango del vehículo
model.range_constraint = ConstraintList()
for v in model.V:
    model.range_constraint.add(
        sum(model.dist[i, j] * model.x[i, j, v] for (i, j) in model.A) <= model.range[v]
    )

# Resolver el modelo
opt = SolverFactory('gurobi')
opt.options['TimeLimit'] = 1800  # Establecer límite de tiempo en segundos
opt.options['MIPGap'] = 0.01  # Establecer tolerancia de optimalidad
opt.options['Threads'] = 4  # Número de hilos a usar
results = opt.solve(model, tee=True)

# Verificar si se encontró una solución factible
if (results.solver.termination_condition == TerminationCondition.maxTimeLimit and
    results.solver.status == SolverStatus.ok):
    print("Tiempo agotado, pero se encontró una solución factible.")

# Acceder a los valores de las variables
for v in model.component_objects(Var, active=True):
    for index in v:
        print(f"{v[index].name} = {value(v[index])}")

# Reconstrucción ordenada de rutas
for v in model.V:
    print(f"\nVehículo {v}:")
    ruta = [1]
    actual = 1
    while True:
        next_nodes = [j for (i, j) in model.A if i == actual and model.x[i, j, v].value > 0.5]
        if not next_nodes:
            break
        siguiente = next_nodes[0]
        ruta.append(siguiente)
        actual = siguiente
        if siguiente == 1:
            break
    print("Ruta:", ruta)

routes = {}

for v in model.V:
    print(f"Vehículo {v}:")
    routes[v] = []

    # Construir lista de arcos activos para el vehículo v
    active_arcs = [(i, j) for i in model.N for j in model.N 
                   if i != j and model.x[i, j, v].value > 0.5]
    
    if not active_arcs:
        continue

    # Iniciar la ruta desde el nodo de depósito (asumimos que es el nodo 1)
    actual = next(i for (i, j) in active_arcs if i in model.D)  # o usar actual = 1 si sabes que siempre empieza ahí
    ruta = [actual]

    while True:
        siguientes = [j for (i, j) in active_arcs if i == actual]
        if not siguientes:
            break
        siguiente = siguientes[0]
        ruta.append(siguiente)
        actual = siguiente
        if siguiente in model.D:
            break

    # Guardar coordenadas ordenadas
    routes[v] = [coord[n] for n in ruta]

    # Mostrar info de carga
    for idx in range(len(ruta) - 1):
        i, j = ruta[idx], ruta[idx+1]
        if i in model.C:
            capacidad_utilizada = model.demand[i] * model.x[i, j, v].value
            print(f"  Ruta {i} -> {j}: Carga transportada = {capacidad_utilizada}")
        else:
            print(f"  Ruta {i} -> {j} (desde depósito)")



# %%
m = folium.Map(
    location=[4.743359, -74.153536],
    zoom_start=11,
    tiles='Cartodb Positron' 
)

number_of_colors = 9

colors = ['blue', 'green', 'cyan', 'magenta','olive', 'blue', 'orange', 'purple','red' ]
icons = ['blue', 'green', 'lightblue', 'pink','lightgreen', 'blue', 'orange', 'darkpurple','red' ]

for route in routes.keys():
    if len(routes[route]) != 0 :  
        folium.PolyLine(
            routes[route],
            color=colors[route],
            weight=5,
            opacity=0.7,
            tooltip='Vehículo ' + str(route)
        ).add_to(m)

        folium.Marker(routes[route][0], popup="Inicio", icon=folium.Icon(color='black')).add_to(m)
        folium.Marker(routes[route][-1], popup="Llegada V" + str(route), icon=folium.Icon(color=icons[route])).add_to(m)
m



# %%
