# ECU2
En esta práctica utilizamos CUDA en Python con la librería Numba. En este ejercicio el objetivo es entender mejor cómo se organizan los hilos y los bloques dentro de la GPU y cómo se calcula el identificador global de cada hilo.

El programa hace lo siguiente:

Define una configuración de ejecución con una malla (grid) bidimensional de bloques y un número fijo de hilos por bloque.

Dentro del kernel, calcula el identificador del bloque a partir de sus coordenadas en X y Y.

Calcula el número de hilos por bloque y el desplazamiento (offset) que aporta cada bloque.

Obtiene el identificador del hilo dentro del bloque usando las coordenadas de hilo en X y Y.

Combina esta información para obtener el identificador global de cada hilo en toda la malla.

Imprime, para cada hilo, su id global, la posición del bloque, la posición del hilo dentro del bloque y las dimensiones de la malla y del bloque.

# CUDA-lab

## Curso: TAE IA – Introducción a CUDA
## Institución: CINVESTAV
## Profesor: Dr. German Pinedo-Díaz
## Alumno: Dr. Antonio Navarrete Guzmán

Este repositorio contiene una serie de ejercicios para aprender los conceptos básicos de programación en GPU utilizando CUDA con Python y la librería Numba. Cada práctica se encuentra en su propio branch y está organizada para avanzar paso a paso desde ejemplos simples hasta aplicaciones más completas.

Los branches incluidos en el repositorio

ECU1	Primer acercamiento a CUDA. Se copian datos desde CPU a GPU, se ejecuta un kernel sencillo y se devuelven los resultados a la CPU.

ECU2	Identificación de hilos y bloques. Cada hilo imprime su información para comprender la organización del paralelismo en GPU.

ECU3	Serie de ejercicios de comparación de desempeño entre GPU y CPU. Incluye varios kernels separados en archivos: suma de vectores, operaciones matemáticas por elemento, escalamiento de matrices, multiplicación de matrices y filtro de Sobel en imágenes.

Objetivo del repositorio

Explorar y comprender la estructura básica de un programa CUDA, poniendo en práctica los elementos principales:

Preparación de datos en CPU
Transferencia de datos CPU → GPU
Ejecución de kernels en paralelo
Transferencia de resultados GPU → CPU
Validación y medición de tiempos

Las prácticas están diseñadas para servir como base antes de pasar a optimización, uso de memoria compartida y otras estrategias de alto rendimiento.

