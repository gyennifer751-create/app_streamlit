import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# ===============================
# FUNCIONES AUXILIARES
# ===============================
def check_feasible(x, y, constraints):
    for (a1, a2, sign, b) in constraints:
        val = a1*x + a2*y
        if sign == "<=" and val > b + 1e-6:
            return False
        if sign == ">=" and val < b - 1e-6:
            return False
        if sign == "=" and abs(val - b) > 1e-6:
            return False
    return True

def evaluate_obj(x, y, c_obj):
    return c_obj[0]*x + c_obj[1]*y

# ===============================
# INTERFAZ STREAMLIT
# ===============================
st.title("Modelador de Programación Lineal - 2 Variables")

opt_type = st.radio("Tipo de optimización", ["max", "min"])

st.subheader("Función objetivo")
a_obj = st.number_input("Coeficiente de x", value=1.0, step=0.5)
b_obj = st.number_input("Coeficiente de y", value=1.0, step=0.5)
c_obj = [a_obj, b_obj]

st.subheader("Restricciones")
num_restricciones = st.number_input("Número de restricciones", min_value=1, max_value=10, value=3)

constraints = []
for i in range(num_restricciones):
    st.markdown(f"### Restricción {i+1}")
    a1 = st.number_input(f"Coeficiente de x (R{i+1})", value=1.0, step=0.5, key=f"a1_{i}")
    a2 = st.number_input(f"Coeficiente de y (R{i+1})", value=1.0, step=0.5, key=f"a2_{i}")
    sign = st.selectbox(f"Tipo de inecuación (R{i+1})", ["<=", ">=", "="], key=f"sign_{i}")
    b = st.number_input(f"Valor (R{i+1})", value=10.0, step=1.0, key=f"b_{i}")
    constraints.append((a1, a2, sign, b))

if st.button("Resolver y graficar"):
    # ===============================
    # CALCULO DE PUNTOS DE INTERSECCION
    # ===============================
    points = []
    for (c1, c2) in combinations(constraints, 2):
        (a1, b1, s1, d1) = c1
        (a2, b2, s2, d2) = c2
        A = np.array([[a1, b1], [a2, b2]])
        B = np.array([d1, d2])
        if np.linalg.matrix_rank(A) == 2:
            sol = np.linalg.solve(A, B)
            x, y = sol
            if x >= -1e-6 and y >= -1e-6:
                if check_feasible(x, y, constraints):
                    points.append((x, y))

    # ===============================
    # EVALUAR EN VERTICES
    # ===============================
    best_val = None
    best_point = None

    for (x, y) in points:
        val = evaluate_obj(x, y, c_obj)
        if best_val is None:
            best_val, best_point = val, (x, y)
        else:
            if opt_type == "max" and val > best_val:
                best_val, best_point = val, (x, y)
            if opt_type == "min" and val < best_val:
                best_val, best_point = val, (x, y)

    # ===============================
    # GRAFICAR
    # ===============================
    x_vals = np.linspace(0, 100, 400)
    y_vals = np.linspace(0, 100, 400)

    fig, ax = plt.subplots(figsize=(8,6))

    for (a1, a2, sign, b) in constraints:
        if a2 != 0:
            y_line = (b - a1*x_vals)/a2
            ax.plot(x_vals, y_line, label=f"{a1}x + {a2}y {sign} {b}")
        else:
            x_line = np.full_like(y_vals, b/a1)
            ax.plot(x_line, y_vals, label=f"{a1}x + {a2}y {sign} {b}")

    feasible_x = [p[0] for p in points]
    feasible_y = [p[1] for p in points]
    ax.scatter(feasible_x, feasible_y, color="blue", zorder=5, label="Vértices factibles")

    if best_point:
        ax.scatter(best_point[0], best_point[1], color="red", s=100, zorder=6, label=f"Óptimo: {best_point}, Z={best_val:.2f}")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title("Modelo de Programación Lineal - 2 Variables")
    ax.grid(True)

    st.pyplot(fig)

    st.subheader("Resultados")
    st.write("Vértices factibles evaluados:")
    for (x, y) in points:
        st.write(f"({x:.2f}, {y:.2f}) -> Z = {evaluate_obj(x,y,c_obj):.2f}")

    if best_point:
        st.success(f"Óptimo encontrado: {best_point} con Z = {best_val:.2f}")
