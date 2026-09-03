# Aprendizaje Supervisado

Repositorio de materiales, ejemplos y notebooks de la asignatura **Aprendizaje Supervisado** del Máster en Inteligencia Artificial de la Universidad Internacional de Valencia (VIU).

El proyecto reúne ejercicios prácticos sobre preparación de datos, evaluación de modelos, tratamiento de variables, visión artificial, regresión y clasificación.

## Contenidos

| Carpeta | Temas principales |
| --- | --- |
| `tema1/` | Terminología, limpieza de datos, imputación y selección de características |
| `tema2/` | Métricas y validación de modelos |
| `tema3/` | Variables cuantitativas y cualitativas, transformaciones de imágenes, segmentación y descriptores |
| `tema4/` | Regresión lineal, regularización y regresión polinómica |
| `tema5/` | Regresión logística, árboles de decisión y máquinas de vectores soporte (SVM) |
| `datasets/` | Conjuntos de datos utilizados en las prácticas |
| `examen/` | Recursos gráficos relacionados con ejercicios de evaluación |
| `logos/` | Imágenes y recursos utilizados por los notebooks |

Algunos temas contienen una versión base y una `version_clase` con el desarrollo realizado durante las sesiones.

## Tecnologías

- Python 3
- Jupyter Notebook
- NumPy y pandas
- Matplotlib y seaborn
- scikit-learn
- SciPy y scikit-image
- OpenCV
- Graphviz

## Instalación

Clona el repositorio y entra en su directorio:

```bash
git clone https://github.com/ssanchezgoe/viu_aprendizaje_supervisado.git
cd viu_aprendizaje_supervisado
```

Se recomienda crear un entorno virtual antes de instalar las dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyter numpy pandas matplotlib seaborn scikit-learn scipy scikit-image opencv-python graphviz mglearn tabulate termcolor
```

> Para visualizar árboles con Graphviz también puede ser necesario instalar su ejecutable en el sistema operativo.

## Uso

Inicia Jupyter desde la raíz del repositorio:

```bash
jupyter notebook
```

Después, abre los notebooks siguiendo el orden de los temas. Ejecutarlos desde la raíz ayuda a conservar las rutas relativas hacia `datasets/`, `logos/` y `tema3/imagenes/`.

## Datos

El repositorio incluye, entre otros, los conjuntos de datos `penguins.csv` y `outliers.csv`, empleados en los ejemplos de exploración, preparación y modelado.

## Finalidad

Este repositorio tiene fines educativos y sirve como material de apoyo para estudiar y practicar técnicas de aprendizaje supervisado.
