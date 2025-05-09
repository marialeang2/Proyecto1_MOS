# Proyecto MOS - Entrega 2 - Implementación

## Proyecto A

##  Integrantes

- Laura Sofia Murcia  Carreño — 202123099
- María Alejandra Angulo - 202121458S
- Samuel Ramirez Gomez - 202123423

## Pasos para correr el proyecto

Revise que su ambiente de ejecución cuenta con los paquetes definidos dentro del archivo de requirements. Adicionalmente, para la ejecución correcta de los modelos se utilizo la licensia estudiantil de `Gurobi`, por lo que asegurese de tener una versión de esta activa al momento de correr el proyecto. Asimismo, para agilizar la ejecución de los modelos se utiliza el parámetro Threads de `Gurobi`. Está fijo a 4 threads, confirme que su máquina puede correr 4 threads simultáneos. De no ser posible, ponga el máximo de threads que puede operar su máquina. 

## Estructura del repositorio

El proyecto se divide en 3 carpetas: `Datos`,  `Modelos` y `Documentos_base`. 

### Documentos base
Dentro de la carpeta de documentos base se encuentra el enunciado general de esta iteración del proyecto realizado y el documento de la entrega de modelado matemático para su consulta fácil. 

### Datos 
Dentro de la carpeta de `Datos` se encuentran todos los archivos referentes a la información de cada caso. Esta carpeta se divide en las carpetas: API, Caso_Base, Caso2 y Caso3. 

En la carpeta API se tiene un archivo para ejecutar un servidor con conexión a la API OSRM la cual fue utilizada para calcular las distancias y tiempos entre depositos y clientes. Para ejecutar los modelos NO es necesario ejecutar el servidor, pero se incluye como recurso que fue utilizado en la realización del proyecto. 

En las carpetas Caso_Base, Caso2 y Caso3 se encuentran los archivos con los datos que se brindaron de clientes, depositos y vehiculos dependiendo el caso. Adicionalmente, se encuentra un archivo `.csv` generado con los datos de distancia y tiempo entre cada par de nodos dependiendo el caso. Este archivo se creo con el código dentro del archivo data.ipynb que se encuentra dentro de la carpeta de Datos. La explicación de la generación de estos datos se puede encontrar dentro de ese archivo. 

### Modelos

Dentro de la carpeta de modelos se encuentran 3 notebooks en donde se desarrolla la solución y se incluye el análisis y visualización de cada uno de los casos resueltos. Adicionalmente, en esta carpeta se incluyen los archivos de verificación para cada caso. Estos archivos de verificación aparecerán como .csv en la misma carpeta de `Modelos`.

