#ECU1
Este ejemplo corresponde a mis primeras prácticas utilizando CUDA en Python con la librería Numba. El propósito principal es comenzar a familiarizarme con el procesamiento en GPU y entender la estructura básica que se requiere para ejecutar un kernel.

El programa hace lo siguiente:

• Genera datos en la CPU.
• Copia esos datos a la GPU.
• Ejecuta un kernel sencillo en la GPU que suma elemento por elemento los valores de dos arreglos.
• Recupera los resultados en la CPU.
• Imprime parte del resultado y el tiempo de ejecución.
