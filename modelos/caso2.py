# %%
import numpy as np
from pyomo.environ import *
import pandas as pd
import folium
import re

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
distancia,time_dic = load_distance_time_dic('../Datos/Caso2/caso2.csv')
vehiculos = load_vehicles('../Datos/Caso2/vehicles.csv')
demanda = load_demand('..\Datos\Caso2\clients.csv')
oferta = load_capacity('..\Datos\Caso2\depots.csv')
coord = load_coordinates('..\Datos\Caso2\depots.csv', '..\Datos\Caso2\clients.csv')

print(f"Distancia: {distancia}")
print(f"Vehiculos: {vehiculos}")
print(f"Oferta: {oferta}")
print(f"Demanda: {demanda}")
print(f"Coordenadas: {coord}")
print(f"Time: {time_dic}")



# %%
pf = 15000
ft = 5000
cm = 700
gv = 0.411458


costo_transporte = gv*pf +ft+cm

# Modelo
model = ConcreteModel()

# Conjuntos
model.D = Set(initialize=oferta.keys())   # Depósitos
model.C = Set(initialize=demanda.keys())  # Clientes
model.N = Set(initialize=list(model.D) + list(model.C))  # Todos los nodos
model.K = Set(initialize=vehiculos.keys())  # Vehículos

# Parámetros
model.distancia = Param(model.N, model.N, initialize=lambda model, i, j: distancia.get((i,j), 0))
model.capacidad = Param(model.K, initialize=lambda model, k: vehiculos[k][0])
model.demanda = Param(model.C, initialize=demanda)
model.oferta = Param(model.D, initialize=oferta)

# Variables
model.x = Var(model.N, model.N, model.K, within=Binary)  # Arco i-j servido por vehículo k
model.y = Var(model.C, model.K, within=NonNegativeReals) # Cantidad entregada a cliente j por vehículo k
model.u = Var(model.C, within=NonNegativeIntegers)       # Variables para eliminar subciclos

pf = 15000
ft = 5000
cm = 700
gv = 0.411458


costo_transporte = gv*pf +ft+cm

# Función objetivo
def obj_rule(model):
    return sum(costo_transporte*model.distancia[i,j] * model.x[i,j,k] for i in model.N for j in model.N for k in model.K if i != j)
model.obj = Objective(rule=obj_rule, sense=minimize)

# Restricciones
def demanda_satisfecha_rule(model, j):
    return sum(model.y[j,k] for k in model.K) == model.demanda[j]
model.demanda_satisfecha = Constraint(model.C, rule=demanda_satisfecha_rule)

def capacidad_vehiculo_rule(model, k):
    return sum(model.y[j,k] for j in model.C) <= model.capacidad[k]
model.capacidad_vehiculo = Constraint(model.K, rule=capacidad_vehiculo_rule)

def flujo_entrada_salida_rule(model, j, k):
    return sum(model.x[i,j,k] for i in model.N if i != j) == sum(model.x[j,i,k] for i in model.N if i != j)
model.flujo_entrada_salida = Constraint(model.N, model.K, rule=flujo_entrada_salida_rule)

def salida_deposito_rule(model, d, k):
    return sum(model.x[d,j,k] for j in model.C) <= 1  # Máximo 1 ruta que sale del depósito d por vehículo k
model.salida_deposito = Constraint(model.D, model.K, rule=salida_deposito_rule)

def subtour_elimination_rule(model, i, j):
    if i != j and i in model.C and j in model.C:
        return model.u[i] - model.u[j] + len(model.C)*sum(model.x[i,j,k] for k in model.K) <= len(model.C) - 1
    else:
        return Constraint.Skip
model.subtour_elimination = Constraint(model.N, model.N, rule=subtour_elimination_rule)

def asignacion_unica_rule(model, j):
    if j in model.C:
        return sum(model.x[i,j,k] for i in model.N for k in model.K if i != j) == 1
    else:
        return Constraint.Skip
model.asignacion_unica = Constraint(model.N, rule=asignacion_unica_rule)

# Resolver
solver = SolverFactory('gurobi') 

# Parámetros importantes:
solver.options['Threads'] = 4  # Número de hilos a usar (ajusta según tu CPU)
results = solver.solve(model, tee=False)  # tee=True muestra el progreso en consola
# Mostrar resultados
if results.solver.termination_condition == TerminationCondition.optimal:
    print("Solución óptima encontrada")
    print("Valor de la función objetivo:", model.obj())
    
    routes_dict = {}  # Guardamos las rutas en coordenadas para folium

    for v in model.K:
        print(f"\nVehículo {v}:")
        
        active_arcs = [(i,j) for i in model.N for j in model.N 
                    if i != j and model.x[i,j,v].value > 0.5]
        
        if not active_arcs:
            print("  No utilizado")
            routes_dict[v] = []
            continue
        
        route_nodes = []
        current = next((i for i,j in active_arcs if i in model.D), active_arcs[0][0])
        route_nodes.append(current)
        remaining_arcs = active_arcs.copy()

        while remaining_arcs:
            next_arc = next((a for a in remaining_arcs if a[0] == current), None)
            if next_arc:
                current = next_arc[1]
                route_nodes.append(current)
                remaining_arcs.remove(next_arc)
            else:
                break

        # Mostrar ruta ordenada
        print(" -> ".join(str(node) for node in route_nodes))

        # Mostrar demanda atendida
        for i in range(len(route_nodes)-1):
            origen = route_nodes[i]
            destino = route_nodes[i+1]
            if destino in model.C:
                print(f"  {origen}->{destino}: Atiende cliente {destino} (Demanda: {model.demanda[destino]})")
            else:
                print(f"  {origen}->{destino}")

        # Guardar coordenadas de la ruta para folium
        if 'coord' in globals():
            routes_dict[v] = [coord[nodo] for nodo in route_nodes]
        else:
            print("Falta el diccionario 'coordenadas' para graficar en folium.")
            routes_dict[v] = []


else:
    print("No se encontró solución óptima")
    print("Estado del solver:", results.solver.termination_condition)

# %% [markdown]
# ## Archivo de validación 
# 

# %%
data = {
    'VehicleId':[] ,
    'DepotId':[] ,
    'InitialLoad':[] ,
    'RouteSequence':[] ,
    'ClientsServed':[] ,
    'DemandsSatisfied':[] , 
    'TotalDistance':[] , 
    'TotalTime':[] , 
    'FuelCost':[]
}

def isClient(name):
    return bool(re.fullmatch(r'C\d+', name))

for v in model.K:
    path = []
     
    
    active_arcs = [(i,j) for i in model.N for j in model.N 
                if i != j and model.x[i,j,v].value > 0.5]
    
    if not active_arcs:
        routes_dict[v] = []
        continue
    
    data['VehicleId'].append("VEH" + "{:03d}".format(v))
    route_nodes = []
    current = next((i for i,j in active_arcs if i in model.D), active_arcs[0][0])
    if current in model.C:
        name = "C" + str(current)
    else:
        name = "CD" + str(current)
        data['DepotId'].append(name)
    route_nodes.append(name)
    remaining_arcs = active_arcs.copy()


    while remaining_arcs:
        next_arc = next((a for a in remaining_arcs if a[0] == current), None)
        if next_arc:
            current = next_arc[1]
            if current in model.C:
                name = "C" + str(current)
            else:
                name = "CD" + str(current)
            route_nodes.append(name)
            remaining_arcs.remove(next_arc)
        else:
            break
    
    total_dist = 0
    t_time = 0
    for i, j in active_arcs:
        total_dist += distancia[(i,j)]
        t_time += time_dic[(i,j)]

    data['TotalDistance'].append(round(total_dist/1000,1))
    data['TotalTime'].append(round(t_time, 1))
    data['FuelCost'].append(round((total_dist/1000)*pf))
    print("time", t_time)

    seq = "-".join(str(node) for node in route_nodes)
    print(seq)
    clientsServe = 0
    demandsS = []
    init_load = 0
    for node in route_nodes:
        if isClient(node):
            clientsServe += 1
            demandsS.append(demanda[int(node.split('C')[1].strip())])
            init_load += demanda[int(node.split('C')[1].strip())]
    #print(clientsServe)
    print(demandsS)
    dema = "-".join(str(d) for d in demandsS)
    #print(seq)
    data['RouteSequence'].append(seq)
    data['ClientsServed'].append(clientsServe)
    data['DemandsSatisfied'].append(dema)
    data['InitialLoad'].append(init_load)   

print(data)
df = pd.DataFrame(data)
df.to_csv("verificacion_caso2.csv", index=False)
    
    

#print(data)

# %%
import folium

m = folium.Map(
    location=[4.743359, -74.153536],  # Puedes centrarlo dinámicamente si gustas
    zoom_start=12,
    tiles='Cartodb Positron'
)

colors = ['blue', 'green', 'cyan', 'magenta', 'olive', 'orange', 'purple', 'red', 'darkred']
icons = ['blue', 'green', 'lightblue', 'pink', 'lightgreen', 'orange', 'darkpurple', 'red', 'darkred']

for v, coords in routes_dict.items():
    if len(coords) > 1:
        folium.PolyLine(
            coords,
            color=colors[int(v) % len(colors)],
            weight=5,
            opacity=0.8,
            tooltip=f"Vehículo {v}"
        ).add_to(m)

        folium.Marker(
            location=coords[0],
            popup=f"Inicio V{v}",
            icon=folium.Icon(color='black')
        ).add_to(m)

        folium.Marker(
            location=coords[-1],
            popup=f"Fin V{v}",
            icon=folium.Icon(color=icons[int(v) % len(icons)])
        ).add_to(m)
m




# %%
