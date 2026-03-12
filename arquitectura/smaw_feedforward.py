import sys

sys.path.append("/home/luis/PlotNeuralNet/")
from pycore.tikzeng import *


def to_Aggregator(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=3,
    height=30,
    depth=30,
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
        fill=\PoolColor,
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


def to_FCBlock(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=3,
    height=30,
    depth=30,
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
        fill=\FcColor,
        bandfill=\FcReluColor,
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
    # YAMNet: extractor pre-entrenado
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
    # Aggregator: mean + std -> 2048
    to_Aggregator(
        name="aggregator",
        offset="(2.5,0,0)",
        to="(yamnet-east)",
        width=3,
        height=32,
        depth=32,
        caption=r"{\parbox{2.4cm}{\centering\small\textbf{Aggregator}\\\footnotesize mean + std\\\footnotesize 1$\times$2048}}",
    ),
    to_connection("yamnet", "aggregator"),
    # FC-1: 2048 -> 1024, BN + ReLU + Drop
    to_FCBlock(
        name="fc1",
        offset="(2.5,0,0)",
        to="(aggregator-east)",
        width=3.2,
        height=36,
        depth=36,
        caption=r"{\parbox{2.8cm}{\centering\small\textbf{FC-1}\\\footnotesize 2048$\times$1024\\\footnotesize BN+ReLU+Drop}}",
    ),
    to_connection("aggregator", "fc1"),
    # FC-2: 1024 -> 512, BN + ReLU + Drop
    to_FCBlock(
        name="fc2",
        offset="(2.5,0,0)",
        to="(fc1-east)",
        width=2.8,
        height=32,
        depth=32,
        caption=r"{\parbox{2.8cm}{\centering\small\textbf{FC-2}\\\footnotesize 1024$\times$512\\\footnotesize BN+ReLU+Drop}}",
    ),
    to_connection("fc1", "fc2"),
    # FC-3: 512 -> 256, BN + ReLU + Drop
    to_FCBlock(
        name="fc3",
        offset="(2.5,0,0)",
        to="(fc2-east)",
        width=2.5,
        height=28,
        depth=28,
        caption=r"{\parbox{2.8cm}{\centering\small\textbf{FC-3}\\\footnotesize 512$\times$256\\\footnotesize BN+ReLU+Drop}}",
    ),
    to_connection("fc2", "fc3"),
    # Head: Espesor (256, 3)
    to_Head(
        name="head_espesor",
        offset="(4.0,4.5,0)",
        to="(fc3-east)",
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
        to="(fc3-east)",
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
        to="(fc3-east)",
        width=2.8,
        height=8,
        depth=8,
        opacity=0.9,
        caption=r"{\parbox{2.4cm}{\centering\small\textbf{Corriente}\\\footnotesize 256$\times$2}}",
    ),
    # Conexiones a los heads
    r"""\draw [connection]  (fc3-east) -- node {\midarrow} (head_espesor-west);""",
    r"""\draw [connection]  (fc3-east) -- node {\midarrow} (head_electrodo-west);""",
    r"""\draw [connection]  (fc3-east) -- node {\midarrow} (head_corriente-west);""",
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
