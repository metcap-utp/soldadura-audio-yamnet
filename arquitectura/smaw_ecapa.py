import sys

sys.path.append("/home/luis/PlotNeuralNet/")
from pycore.tikzeng import *


def lcap(name, dims, w="2.2"):
    return (r"{\parbox{" + w + r"cm}{\centering\large\textbf{" + name
            + r"}\\\normalsize\textbf{" + dims + "}}}")


def to_ResBlock(name, offset="(0,0,0)", to="(0,0,0)", width=3, height=35, depth=35, caption=" "):
    return (r"""
\pic[shift={""" + offset + """}] at """ + to + """
    {RightBandedBox={
        name=""" + name + """,
        caption=""" + caption + """,
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height=""" + str(height) + """,
        width=""" + str(width) + """,
        depth=""" + str(depth) + """
        }
    };
""")


def to_ASP(name, offset="(0,0,0)", to="(0,0,0)", width=3, height=35, depth=35, caption=" "):
    return (r"""
\pic[shift={""" + offset + """}] at """ + to + """
    {Box={
        name=""" + name + """,
        caption=""" + caption + """,
        fill=\PoolColor,
        height=""" + str(height) + """,
        width=""" + str(width) + """,
        depth=""" + str(depth) + """
        }
    };
""")


def to_FC(name, offset="(0,0,0)", to="(0,0,0)", width=2, height=20, depth=20, caption=" "):
    return (r"""
\pic[shift={""" + offset + """}] at """ + to + """
    {Box={
        name=""" + name + """,
        caption=""" + caption + """,
        fill=\FcColor,
        height=""" + str(height) + """,
        width=""" + str(width) + """,
        depth=""" + str(depth) + """
        }
    };
""")


def to_BN(name, offset="(0,0,0)", to="(0,0,0)", width=2.2, height=32, depth=32, caption=" "):
    return (r"""
\pic[shift={""" + offset + """}] at """ + to + """
    {Box={
        name=""" + name + """,
        caption=""" + caption + """,
        fill=\FcColor,
        height=""" + str(height) + """,
        width=""" + str(width) + """,
        depth=""" + str(depth) + """
        }
    };
""")


def to_Head(name, offset="(0,0,0)", to="(0,0,0)", width=2.8, height=10, depth=10, opacity=0.9, caption=" "):
    return (r"""
\pic[shift={""" + offset + """}] at """ + to + """
    {Box={
        name=""" + name + """,
        caption=""" + caption + """,
        fill=\SoftmaxColor,
        opacity=""" + str(opacity) + """,
        height=""" + str(height) + """,
        width=""" + str(width) + """,
        depth=""" + str(depth) + """
        }
    };
""")


COLORDEFS = r"""
\usetikzlibrary{calc}
\definecolor{LegConv}{RGB}{255,204,102}
\definecolor{LegConvRelu}{RGB}{255,170,85}
\definecolor{LegPool}{RGB}{196,0,0}
\definecolor{LegFc}{RGB}{153,102,204}
\definecolor{LegFcRelu}{RGB}{164,73,164}
\definecolor{LegSoftmax}{RGB}{106,0,106}
\definecolor{LegEdge}{RGB}{32,128,128}
"""

LEGEND_ENG = r"""
\path (current bounding box.south) coordinate (BBOXS);
\begin{scope}[shift={($(BBOXS)+(-9.8,-3.5)$)}]
  \node[anchor=west,font=\Large\bfseries] at (0,2.0) {Legend};
  \fill[LegConv] (0,0) rectangle (2.0,0.55);
  \node[anchor=west,font=\Large] at (2.4,0.275) {Pre-trained extractor};
  \fill[LegConv] (0,-0.9) rectangle (1.7,-0.35); \fill[LegConvRelu] (1.7,-0.9) rectangle (2.0,-0.35);
  \node[anchor=west,font=\Large] at (2.4,-0.625) {TDNN / Res2Net (Conv + BN + ReLU)};
  \fill[LegPool] (0,-1.8) rectangle (2.0,-1.25);
  \node[anchor=west,font=\Large] at (2.4,-1.525) {Attentive Stats Pooling};
  \fill[LegFc] (0,-2.7) rectangle (2.0,-2.15);
  \node[anchor=west,font=\Large] at (2.4,-2.425) {FC + BN / MFA};
  \fill[LegSoftmax,opacity=0.9] (0,-3.6) rectangle (2.0,-3.05);
  \node[anchor=west,font=\Large] at (2.4,-3.325) {Classification (heads)};
  \draw[-Stealth,line width=1pt,LegEdge] (0,-4.4) -- (2.0,-4.4);
  \node[anchor=west,font=\Large] at (2.4,-4.4) {Data flow};
  \draw[rounded corners=4pt,black,line width=0.6pt] (-0.4,2.5) rectangle (20,-5.0);
\end{scope}
"""

arch = [
    to_head("/home/luis/PlotNeuralNet/"),
    to_cor(),
    COLORDEFS,
    to_begin(),
    r"""\pic[shift={(0,0,0)}] at (0,0,0)
    {Box={
        name=yamnet,
        caption=""" + lcap("YAMNet", r"1$\times$1$\times$1024") + r""",
        fill=\ConvColor,
        height=40,
        width=3,
        depth=40
        }
    };""",
    to_BN("bn_input", "(2.6,0,0)", "(yamnet-east)", 2.2, 32, 32,
          lcap("BatchNorm", "1024")),
    to_connection("yamnet", "bn_input"),
    to_ResBlock("tdnn", "(2.6,0,0)", "(bn_input-east)", 3, 36, 36,
                lcap("TDNN", r"512$\times$1024$\times$5")),
    to_connection("bn_input", "tdnn"),
    to_ResBlock("res2net1", "(2.6,0,0)", "(tdnn-east)", 3, 35, 35,
                lcap("Res2Net-1", r"512$\times$512$\times$3")),
    to_connection("tdnn", "res2net1"),
    to_ResBlock("res2net2", "(2.6,0,0)", "(res2net1-east)", 3, 35, 35,
                lcap("Res2Net-2", r"512$\times$512$\times$3")),
    to_connection("res2net1", "res2net2"),
    to_ResBlock("res2net3", "(2.6,0,0)", "(res2net2-east)", 3, 35, 35,
                lcap("Res2Net-3", r"512$\times$512$\times$3")),
    to_connection("res2net2", "res2net3"),
    to_FC("mfa", "(2.6,0,0)", "(res2net3-east)", 4, 40, 40,
          lcap("MFA", r"1536$\times$1536")),
    to_connection("res2net3", "mfa"),
    to_ASP("asp", "(2.6,0,0)", "(mfa-east)", 3, 35, 35,
           lcap("ASP", "3072", "2.0")),
    to_connection("mfa", "asp"),
    to_FC("fc_embedding", "(2.6,0,0)", "(asp-east)", 2.5, 22, 22,
          lcap("FC + BN", r"3072$\times$256")),
    to_connection("asp", "fc_embedding"),
    to_Head("head_espesor", "(3.5,4.5,0)", "(fc_embedding-east)", 2.8, 10, 10, 0.9, " "),
    to_Head("head_electrodo", "(3.5,0,0)", "(fc_embedding-east)", 2.8, 12, 12, 0.9, " "),
    to_Head("head_corriente", "(3.5,-4.5,0)", "(fc_embedding-east)", 2.8, 8, 8, 0.9, " "),
    r"""\draw [connection]  (fc_embedding-east) -- node {\midarrow} (head_espesor-west);""",
    r"""\draw [connection]  (fc_embedding-east) -- node {\midarrow} (head_electrodo-west);""",
    r"""\draw [connection]  (fc_embedding-east) -- node {\midarrow} (head_corriente-west);""",
        r"""\path (current bounding box.east) coordinate (BB-EAST);
""",
    r"""\node[anchor=west, xshift=8pt, align=center] at (BB-EAST |- head_espesor-east) {""" + lcap("Espesor", r"256$\times$3") + r"""};""",
    r"""\node[anchor=west, xshift=8pt, align=center] at (BB-EAST |- head_electrodo-east) {""" + lcap("Electrodo", r"256$\times$4") + r"""};""",
    r"""\node[anchor=west, xshift=8pt, align=center] at (BB-EAST |- head_corriente-east) {""" + lcap("Corriente", r"256$\times$2") + r"""};""",
    r"""
\path (current bounding box.south) coordinate (BBOXS);
\begin{scope}[shift={($(BBOXS)+(-9.8,-3.5)$)}]
  \node[anchor=west,font=\Large\bfseries] at (0,2.0) {Leyenda};
  \fill[LegConv] (0,0) rectangle (2.0,0.55);
  \node[anchor=west,font=\Large] at (2.4,0.275) {Extractor pre-entrenado};
  \fill[LegConv] (0,-0.9) rectangle (1.7,-0.35); \fill[LegConvRelu] (1.7,-0.9) rectangle (2.0,-0.35);
  \node[anchor=west,font=\Large] at (2.4,-0.625) {TDNN / Res2Net (Conv + BN + ReLU)};
  \fill[LegPool] (0,-1.8) rectangle (2.0,-1.25);
  \node[anchor=west,font=\Large] at (2.4,-1.525) {Attentive Stats Pooling};
  \fill[LegFc] (0,-2.7) rectangle (2.0,-2.15);
  \node[anchor=west,font=\Large] at (2.4,-2.425) {FC + BN / MFA};
  \fill[LegSoftmax,opacity=0.9] (0,-3.6) rectangle (2.0,-3.05);
  \node[anchor=west,font=\Large] at (2.4,-3.325) {Clasificacion (cabezas)};
  \draw[-Stealth,line width=1pt,LegEdge] (0,-4.4) -- (2.0,-4.4);
  \node[anchor=west,font=\Large] at (2.4,-4.4) {Flujo de datos};
  \draw[rounded corners=4pt,black,line width=0.6pt] (-0.4,2.5) rectangle (20,-5.0);
\end{scope}
""",
    to_end(),
]


def main():
    base = str(sys.argv[0]).split(".")[0]
    arch_esp = arch[:-1] + [to_end()]
    to_generate(arch_esp, base + "_esp.tex")
    print(f"Generado: {base}_esp.tex")
    arch_eng = arch[:-2] + [LEGEND_ENG, to_end()]
    to_generate(arch_eng, base + "_eng.tex")
    print(f"Generado: {base}_eng.tex")


if __name__ == "__main__":
    main()
