"""
 *
 * Copyright (C) 2025 Juan Solsona
 *                    Josè Luis Salazar
 *                    Josè David Gòmez
 *
"""
import math
from typing import List, Dict

from python.normalizar import DatosEjemplo


# Clase que define una neurona
class Neurona:
    def __init__(self, nombre: str, n_entradas: int):
        self.nombre = nombre
        self.n_entradas = n_entradas

        # Pesos y sesgo iniciales
        self.w: List[float] = [0.5 for _ in range(n_entradas)]
        self.b: float = 0.5

        # Valores intermedios
        self.h: float | None = None
        self.z: float | None = None  # salida sigmoide

        # Gradientes
        self.grad_w: List[float] = [0.0 for _ in range(n_entradas)]
        self.grad_b: float = 0.0

    def progresa_valor(self, x: List[float]) -> float:
        """Forward de una neurona: h = w·x + b, z = sigma(h)."""
        h = 0.0
        for i in range(len(self.w)):
            h += self.w[i] * x[i]
        h += self.b

        self.h = h
        self.z = 1.0 / (1.0 + math.exp(-h))
        return self.z


class Capa:
    def __init__(self, n_entradas: int, n_neuronas: int, nombre: str = ""):
        self.nombre = nombre
        self.neuronas: List[Neurona] = [
            Neurona(nombre=f"{nombre}_n{i+1}", n_entradas=n_entradas)
            for i in range(n_neuronas)
        ]

    def progresa_valor(self, x: List[float]) -> List[float]:
        return [neurona.progresa_valor(x) for neurona in self.neuronas]

class Red:
    def __init__(self):
        self.oculta = Capa(2, 3, nombre="oculta")
        self.salida = Capa(3, 1, nombre="salida")

        # Historial para 3.2 y 3.3
        self.historial_entrenamiento: list[dict] = []
        self.predicciones_test: list[dict] = []

    def progresa_valor(self, x):
        z = self.oculta.progresa_valor(x)
        yhat = self.salida.progresa_valor(z)[0]
        return z, yhat

    def backprop(self, x: list[float], y_real: float, eta: float):
        # 1. Forward
        z, yhat = self.progresa_valor(x)

        # --- CAPA DE SALIDA ---
        neurona_salida = self.salida.neuronas[0]

        # 2. Delta de la salida: (ŷ - y) * ŷ * (1-ŷ)
        delta_out = (yhat - y_real) * yhat * (1.0 - yhat)

        # 3. Gradientes de la neurona de salida
        for j in range(len(neurona_salida.w)):
            neurona_salida.grad_w[j] = delta_out * z[j]
        neurona_salida.grad_b = delta_out

        # --- CAPA OCULTA ---
        for j, neurona_oculta in enumerate(self.oculta.neuronas):
            v_j = neurona_salida.w[j]
            z_j = neurona_oculta.z

            delta_j = delta_out * v_j * z_j * (1.0 - z_j)

            for i in range(len(neurona_oculta.w)):
                neurona_oculta.grad_w[i] = delta_j * x[i]

            neurona_oculta.grad_b = delta_j

        # Actualización pesos salida
        for j in range(len(neurona_salida.w)):
            neurona_salida.w[j] -= eta * neurona_salida.grad_w[j]
        neurona_salida.b -= eta * neurona_salida.grad_b

        # Actualización pesos capa oculta
        for neurona_oculta in self.oculta.neuronas:
            for i in range(len(neurona_oculta.w)):
                neurona_oculta.w[i] -= eta * neurona_oculta.grad_w[i]
            neurona_oculta.b -= eta * neurona_oculta.grad_b

        error = 0.5 * (yhat - y_real) ** 2
        return error, yhat

    def entrenar(self, datos_mapa: dict[int, DatosEjemplo], eta: float, n_epocas: int):
        self.historial_entrenamiento = []

        for epoca in range(n_epocas):
            error_total = 0.0
            print(f"\n Época {epoca + 1} ")

            for lote_id in sorted(datos_mapa.keys()):
                dato = datos_mapa[lote_id]
                if not dato.es_entrenamiento:
                    continue

                x1 = dato.edad
                x2 = dato.ingresos
                x = [x1, x2]
                y_real = 1.0 if dato.compra else 0.0

                error, yhat = self.backprop(x, y_real, eta)
                error_total += error

                print(f"Lote {lote_id:2d} | x={x} | y_real={y_real:.1f} | "
                      f"ŷ={yhat:.4f} | e={error:.6f}")

            print(f"Error total en la época {epoca + 1}: {error_total:.6f}")

        # Al terminar la última época, guardamos el estado de la red lote a lote
        # (usando los datos de entrenamiento).
        for lote_id in sorted(datos_mapa.keys()):
            dato = datos_mapa[lote_id]
            if not dato.es_entrenamiento:
                continue

            x1 = dato.edad
            x2 = dato.ingresos
            x = [x1, x2]
            _, yhat = self.progresa_valor(x)

            n1 = self.oculta.neuronas[0]
            n2 = self.oculta.neuronas[1]
            n3 = self.oculta.neuronas[2]
            ns = self.salida.neuronas[0]

            fila = {
                "lote": lote_id,
                "x1": x1,
                "x2": x2,
                "y": int(dato.compra),

                "w11": n1.w[0],
                "w12": n1.w[1],
                "b1": n1.b,

                "w21": n2.w[0],
                "w22": n2.w[1],
                "b2": n2.b,

                "w31": n3.w[0],
                "w32": n3.w[1],
                "b3": n3.b,

                "v1": ns.w[0],
                "v2": ns.w[1],
                "v3": ns.w[2],
                "b_out": ns.b,

                "yhat": yhat
            }
            self.historial_entrenamiento.append(fila)

    def predecir_test(self, datos_mapa: dict[int, DatosEjemplo]):
        """
        Rellena self.predicciones_test con los lotes de test (no entrenamiento).
        """
        self.predicciones_test = []

        for lote_id in sorted(datos_mapa.keys()):
            dato = datos_mapa[lote_id]
            if dato.es_entrenamiento:
                continue

            x1 = dato.edad
            x2 = dato.ingresos
            x = [x1, x2]
            _, yhat = self.progresa_valor(x)
            pred = 1 if yhat >= 0.5 else 0

            self.predicciones_test.append({
                "lote": lote_id,
                "x1": x1,
                "x2": x2,
                "yhat": yhat,
                "pred": pred
            })

    def generar_tabla_entrenamiento_latex(self):
        """
        Usa self.historial_entrenamiento y escupe una tabla LaTeX con columnas limpias.
        """
        print("\n=== Tabla LaTeX para el ejercicio 3.2 (entrenamiento) ===\n")

        print(r"\begin{table}[H]")
        print(r"\centering")
        print(r"\scriptsize")
        print(r"\setlength{\tabcolsep}{3pt}")
        print(r"\begin{tabular}{|c|cc|c|ccc|ccc|ccc|cccc|c|}")
        print(r"\hline")
        print(r"Lote & $x_1$ & $x_2$ & $y$ & "
              r"$w_{11}$ & $w_{12}$ & $b_1$ & "
              r"$w_{21}$ & $w_{22}$ & $b_2$ & "
              r"$w_{31}$ & $w_{32}$ & $b_3$ & "
              r"$v_1$ & $v_2$ & $v_3$ & $b$ & $\hat y$ \\")
        print(r"\hline")

        for fila in self.historial_entrenamiento:
            print(
                f"{fila['lote']} & "
                f"{fila['x1']:.4f} & {fila['x2']:.4f} & {fila['y']} & "
                f"{fila['w11']:.4f} & {fila['w12']:.4f} & {fila['b1']:.4f} & "
                f"{fila['w21']:.4f} & {fila['w22']:.4f} & {fila['b2']:.4f} & "
                f"{fila['w31']:.4f} & {fila['w32']:.4f} & {fila['b3']:.4f} & "
                f"{fila['v1']:.4f} & {fila['v2']:.4f} & {fila['v3']:.4f} & "
                f"{fila['b_out']:.4f} & {fila['yhat']:.4f} \\\\"
            )

        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\caption{Evolución de la red durante el entrenamiento.}")
        print(r"\end{table}")

    def generar_tabla_predicciones_latex(self):
        """
        Usa self.predicciones_test para generar la tabla LaTeX del 3.3.
        """
        print("\n=== Tabla LaTeX para el ejercicio 3.3 (predicciones) ===\n")

        print(r"\begin{table}[H]")
        print(r"\centering")
        print(r"\small")
        print(r"\begin{tabular}{|c|cc|c|c|}")
        print(r"\hline")
        print(r"Lote & $x_1$ & $x_2$ & $\hat y$ & Predicción \\")
        print(r"\hline")

        for fila in self.predicciones_test:
            print(
                f"{fila['lote']} & "
                f"{fila['x1']:.4f} & {fila['x2']:.4f} & "
                f"{fila['yhat']:.4f} & {fila['pred']} \\\\"
            )

        print(r"\hline")
        print(r"\end{tabular}")
        print(r"\caption{Predicciones de la red para los lotes de test.}")
        print(r"\end{table}")

    def dump(self):
        print(" Capa oculta ")
        for n in self.oculta.neuronas:
            print(f"{n.nombre}: w={n.w}, b={n.b}, h={n.h}, z={n.z}, "
                  f"grad_w={n.grad_w}, grad_b={n.grad_b}")

        print("\n Capa salida ")
        for n in self.salida.neuronas:
            print(f"{n.nombre}: w={n.w}, b={n.b}, h={n.h}, z={n.z}, "
                  f"grad_w={n.grad_w}, grad_b={n.grad_b}")

def main():
    # Inicializamos y normalizamos datos
    datos_mapa = DatosEjemplo.init_con_sample()
    datos_mapa = DatosEjemplo.normalizar(datos_mapa)  # ahora edad/ingresos están normalizados

    print(" Datos cargados (normalizados) ")
    for lote, dato in sorted(datos_mapa.items()):
        print(f"Lote {lote:2d}: edad={dato.edad:.4f}, ingresos={dato.ingresos:.4f}, "
              f"compra={dato.compra}, entreno={dato.es_entrenamiento}")

    # Creamos la red
    red = Red()

    print("\n Pesos iniciales de la red ")
    red.dump()

    # Entrenamos sobre los lotes de entrenamiento
    eta = 0.1
    n_epocas = 10
    red.entrenar(datos_mapa, eta=eta, n_epocas=n_epocas)

    print("\n Pesos de la red después del entrenamiento ")
    red.dump()

    # Predicciones para los lotes de TEST (3.3)
    red.predecir_test(datos_mapa)

    # Tablas LaTeX
    red.generar_tabla_entrenamiento_latex()
    red.generar_tabla_predicciones_latex()


if __name__ == "__main__":
    main()