import sys

sys.path.append("/home/luis/PlotNeuralNet/")
from pycore.tikzeng import *


def to_ResBlock(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=3,
    height=35,
    depth=35,
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


def to_ASP(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=3,
    height=35,
    depth=35,
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


def to_FC(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2,
    height=20,
    depth=20,
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


def to_BN(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2.2,
    height=32,
    depth=32,
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
    # BatchNorm(1024)
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
    # ResBlock-1: 1024->1024, k=5, SE
    to_ResBlock(
        name="resblock1",
        offset="(2.5,0,0)",
        to="(bn_input-east)",
        width=3,
        height=38,
        depth=38,
        caption=r"{\parbox{2.8cm}{\centering\small\textbf{ResBlock-1}\\\footnotesize 1024$\times$1024$\times$5\\\footnotesize SE}}",
    ),
    to_connection("bn_input", "resblock1"),
    # ResBlock-2: 1024->1024, k=3, SE
    to_ResBlock(
        name="resblock2",
        offset="(2.5,0,0)",
        to="(resblock1-east)",
        width=3,
        height=36,
        depth=36,
        caption=r"{\parbox{2.8cm}{\centering\small\textbf{ResBlock-2}\\\footnotesize 1024$\times$1024$\times$3\\\footnotesize SE}}",
    ),
    to_connection("resblock1", "resblock2"),
    # ResBlock-3: 1024->1024, k=1, SE
    to_ResBlock(
        name="resblock3",
        offset="(2.5,0,0)",
        to="(resblock2-east)",
        width=2.8,
        height=34,
        depth=34,
        caption=r"{\parbox{2.8cm}{\centering\small\textbf{ResBlock-3}\\\footnotesize 1024$\times$1024$\times$1\\\footnotesize SE}}",
    ),
    to_connection("resblock2", "resblock3"),
    # ASP: Attentive Stats Pooling, 2048
    to_ASP(
        name="asp",
        offset="(2.8,0,0)",
        to="(resblock3-east)",
        width=3,
        height=32,
        depth=32,
        caption=r"{\parbox{2.8cm}{\centering\small\textbf{ASP}\\\footnotesize Attentive Stats\\\footnotesize 2048}}",
    ),
    to_connection("resblock3", "asp"),
    # FC + ReLU: 2048 -> 256
    to_FC(
        name="fc_embedding",
        offset="(2.5,0,0)",
        to="(asp-east)",
        width=2.5,
        height=22,
        depth=22,
        caption=r"{\parbox{2.4cm}{\centering\small\textbf{FC + ReLU}\\\footnotesize 2048$\times$256}}",
    ),
    to_connection("asp", "fc_embedding"),
    # Head: Espesor (256, 3)
    to_Head(
        name="head_espesor",
        offset="(4.0,4.5,0)",
        to="(fc_embedding-east)",
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
        to="(fc_embedding-east)",
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
        to="(fc_embedding-east)",
        width=2.8,
        height=8,
        depth=8,
        opacity=0.9,
        caption=r"{\parbox{2.4cm}{\centering\small\textbf{Corriente}\\\footnotesize 256$\times$2}}",
    ),
    # Conexiones a los heads
    r"""\draw [connection]  (fc_embedding-east) -- node {\midarrow} (head_espesor-west);""",
    r"""\draw [connection]  (fc_embedding-east) -- node {\midarrow} (head_electrodo-west);""",
    r"""\draw [connection]  (fc_embedding-east) -- node {\midarrow} (head_corriente-west);""",
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
