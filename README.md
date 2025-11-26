# ECU 3
Este branch reúne varias prácticas de CUDA en Python usando Numba orientadas a comparar el comportamiento de la GPU contra la CPU en diferentes tipos de operaciones.

El código incluye los siguientes ejemplos:

Suma de vectores de gran tamaño, donde se compara el tiempo de ejecución del kernel en GPU contra la misma operación hecha con NumPy en CPU.

Un kernel sencillo que realiza una operación matemática por elemento (basada en raíces y potencias) para medir tiempos y ver la diferencia entre GPU y CPU.

Escalamiento de una matriz grande, multiplicando cada elemento por un escalar y comparando de nuevo el tiempo de la GPU frente a la solución en CPU.

Una implementación ingenua de multiplicación de matrices, donde cada hilo calcula un elemento de la matriz resultado y se contrasta el desempeño con la operación equivalente usando NumPy.

Un ejemplo de procesamiento de imágenes con el filtro de bordes de Sobel, aplicado a una imagen de alta resolución. Se calculan los bordes tanto en GPU como con OpenCV en CPU y se comparan los tiempos y la similitud de los resultados.

El objetivo general de este branch es seguir practicando el uso de CUDA, entender mejor cómo se configura la ejecución en la GPU y observar, de manera experimental, en qué casos la GPU ofrece una ganancia de tiempo importante frente al procesamiento tradicional en CPU.


# Para ejecutar cualquiera de los ejemplos es necesario ejecutar el entorno de instalacion que esta en el main 

