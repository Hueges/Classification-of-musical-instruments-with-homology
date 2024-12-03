import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
from gtda.plotting import plot_point_cloud
from gtda.homology import VietorisRipsPersistence
from ripser import ripser
from persim import plot_diagrams
from gtda.time_series import SingleTakensEmbedding
from gtda.plotting import plot_diagram
from tqdm import tqdm
from gtda.diagrams import HeatKernel


#A = []
#path = 'muzika11'
#window_length = 0.4  # 0.05 seconds

A=[]
path='muzika12'

for filename in os.listdir(path):
    if filename.split('.')[-1] == 'wav':
        print(filename)
        sampling_rate, audio_data = wav.read(path + '\\' + filename)

        # Extract a single channel if stereo audio
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]  # Extract the first channel

        A.append(audio_data)


# for i in A:
#     plt.plot(i)
#     plt.xlabel('Time')
#     plt.ylabel('Amplitude')
#     plt.title('Sound Wave of Middle C on Piano')
#     plt.grid()
#     plt.show(block=True)
#     plt.close()



max_embedding_dimension = 3
max_time_delay = 8
stride = 100

embedder1 = SingleTakensEmbedding(
    parameters_type="fixed",
    time_delay=max_time_delay,
    dimension=max_embedding_dimension,
    stride=stride,
)


def fit_embedder(embedder: SingleTakensEmbedding, y: np.ndarray, verbose: bool = True) -> np.ndarray:
    """Fits a Takens embedder and displays optimal search parameters."""
    y_embedded = embedder.fit_transform(y)

    if verbose:
        print(f"Shape of embedded time series: {y_embedded.shape}")
        print(
            f"Dimenzija oblaka tacaka {embedder.dimension_} vremenska zadrska izmedju vrednosti pri pravljenju oblaka {embedder.time_delay_}"
        )

    return y_embedded


d = []
b = []
labels = []
for i, audio_data in enumerate(A):
    if i < 100:
        label = 0
    elif i > 99:
        label = 1

    labels.append(label)

    result = fit_embedder(embedder1, audio_data)
    d.append(result)
    print("Velicina oblaka", result.size)

labels_array = np.array(labels)

b = d
b_reshaped = []

for i in b:
    k = i.reshape(1, *i.shape)
    # assert isinstance(k, object)
    b_reshaped.append(k)


# fig=plot_point_cloud(d[61])
# fig.show()

diagrami = []
VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2], max_edge_length=5)
for i in tqdm(b_reshaped):
    j = VR.fit_transform(i)
   # l=scaler.fit_transform(j)
    diagrami.append(j)




# sigma=0.5,n_bins=60
kernel = HeatKernel()
Heat_diagrams = []

for diagram in tqdm(diagrami):
    heat_diagram = kernel.fit_transform(diagram)[0]
    Heat_diagrams.append(heat_diagram)

np.save("Heat_diagrams2.npy", Heat_diagrams)
np.save("Labele.npy2", labels_array)