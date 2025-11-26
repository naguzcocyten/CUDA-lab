
# ECU2
En esta práctica utilizamos CUDA en Python con la librería Numba. En este ejercicio el objetivo es entender mejor cómo se organizan los hilos y los bloques dentro de la GPU y cómo se calcula el identificador global de cada hilo.

El programa hace lo siguiente:

Define una configuración de ejecución con una malla (grid) bidimensional de bloques y un número fijo de hilos por bloque.

Dentro del kernel, calcula el identificador del bloque a partir de sus coordenadas en X y Y.

Calcula el número de hilos por bloque y el desplazamiento (offset) que aporta cada bloque.

Obtiene el identificador del hilo dentro del bloque usando las coordenadas de hilo en X y Y.

Combina esta información para obtener el identificador global de cada hilo en toda la malla.

Imprime, para cada hilo, su id global, la posición del bloque, la posición del hilo dentro del bloque y las dimensiones de la malla y del bloque.
