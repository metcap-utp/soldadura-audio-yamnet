import sys

sys.path.append("/home/luis/PlotNeuralNet/")
from pycore.tikzeng import *


def to_FC(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2,
    height=20,
    depth=2,
    caption=" ",
):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        fill=\FcColor,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


def to_Conv1D(
    name,
    kernel=5,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=3,
    height=25,
    depth=3,
    caption=" ",
):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {RightBandedBox={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


def to_BN(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2.2,
    height=28,
    depth=28,
    caption="BatchNorm",
):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        fill=\FcColor,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


def to_Head(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2.8,
    height=10,
    depth=10,
    opacity=0.9,
    caption=" ",
):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        fill=\SoftmaxColor,
        opacity="""
        + str(opacity)
        + """,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


arch = [
    to_head("/home/luis/PlotNeuralNet/"),
    to_cor(),
    to_begin(),
    # YAMNet: extractor pre-entrenado, produce (B, T, 1024)
    r"""\pic[shift={(0,0,0)}] at (0,0,0)
    {Box={
        name=yamnet,
        caption={\parbox{2.4cm}{\centering\small\textbf{YAMNet}\\\footnotesize 1$\times$1$\times$1024}},
        fill=\ConvColor,
        height=40,
        width=3,
        depth=40
        }
    };""",
    # BatchNorm1d(1024, affine=False)
    to_BN(
        name="bn_input",
        offset="(2.5,0,0)",
        to="(yamnet-east)",
        width=2.2,
        height=32,
        depth=32,
        caption=r"{\parbox{2.0cm}{\centering\small\textbf{BatchNorm}\\\footnotesize 1024}}",
    ),
    to_connection("yamnet", "bn_input"),
    # Conv1d(1024, 256, k=5) + BN + ReLU
    to_Conv1D(
        name="conv1",
        offset="(2.5,0,0)",
        to="(bn_input-east)",
        width=3,
        height=32,
        depth=32,
        caption=r"{\parbox{2.8cm}{\centering\small\textbf{Conv1D}\\\footnotesize 256$\times$1024$\times$5\\\footnotesize BN + ReLU}}",
    ),
    to_connection("bn_input", "conv1"),
    # Conv1d(256, 256, k=3) + BN + ReLU
    to_Conv1D(
        name="conv2",
        offset="(2.8,0,0)",
        to="(conv1-east)",
        width=3,
        height=30,
        depth=30,
        caption=r"{\parbox{2.8cm}{\centering\small\textbf{Conv1D}\\\footnotesize 256$\times$256$\times$3\\\footnotesize BN + ReLU}}",
    ),
    to_connection("conv1", "conv2"),
    # Conv1d(256, 512, k=3) + BN + ReLU
    to_Conv1D(
        name="conv3",
        offset="(2.8,0,0)",
        to="(conv2-east)",
        width=4,
        height=28,
        depth=28,
        caption=r"{\parbox{2.8cm}{\centering\small\textbf{Conv1D}\\\footnotesize 512$\times$256$\times$3\\\footnotesize BN + ReLU}}",
    ),
    to_connection("conv2", "conv3"),
    # StatsPooling: mean + std -> 1024
    to_Pool(
        name="stats",
        offset="(2.8,0,0)",
        to="(conv3-east)",
        width=5,
        height=24,
        depth=3,
        opacity=0.8,
        caption=r"{\parbox{2.4cm}{\centering\small\textbf{Stats Pool}\\\footnotesize mean + std\\\footnotesize 1$\times$1024}}",
    ),
    to_connection("conv3", "stats"),
    # Linear(1024, 256) + ReLU
    to_FC(
        name="fc_shared",
        offset="(2.8,0,0)",
        to="(stats-east)",
        width=2.8,
        height=20,
        depth=3,
        caption=r"{\parbox{2.4cm}{\centering\small\textbf{FC + ReLU}\\\footnotesize 1024$\times$256}}",
    ),
    to_connection("stats", "fc_shared"),
    # Head: Espesor (256, 3)
    to_Head(
        name="head_espesor",
        offset="(4.0,4.5,0)",
        to="(fc_shared-east)",
        width=2.8,
        height=10,
        depth=10,
        opacity=0.9,
        caption=r"{\parbox{2.4cm}{\centering\small\textbf{Espesor}\\\footnotesize 256$\times$3}}",
    ),
    # Head: Electrodo (256, 4)
    to_Head(
        name="head_electrodo",
        offset="(4.0,0,0)",
        to="(fc_shared-east)",
        width=2.8,
        height=12,
        depth=12,
        opacity=0.9,
        caption=r"{\parbox{2.4cm}{\centering\small\textbf{Electrodo}\\\footnotesize 256$\times$4}}",
    ),
    # Head: Corriente (256, 2)
    to_Head(
        name="head_corriente",
        offset="(4.0,-4.5,0)",
        to="(fc_shared-east)",
        width=2.8,
        height=8,
        depth=8,
        opacity=0.9,
        caption=r"{\parbox{2.4cm}{\centering\small\textbf{Corriente}\\\footnotesize 256$\times$2}}",
    ),
    # Conexiones a los heads
    r"""\draw [connection]  (fc_shared-east) -- node {\midarrow} (head_espesor-west);""",
    r"""\draw [connection]  (fc_shared-east) -- node {\midarrow} (head_electrodo-west);""",
    r"""\draw [connection]  (fc_shared-east) -- node {\midarrow} (head_corriente-west);""",
    to_end(),
]


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")
    print(f"Archivo generado: {namefile}.tex")
    print(
        f"Para compilar: cd {'/'.join(namefile.split('/')[:-1])} && pdflatex {namefile.split('/')[-1]}.tex"
    )


if __name__ == "__main__":
    main()
