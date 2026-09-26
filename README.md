# Deep-Learning-Project

Il presente lavoro descrive la progettazione, l’implementazione e la valutazione sperimentale
di una pipeline di deep learning per la previsione probabilistica del traffico urbano su reti
di sensori stradali, con particolare attenzione alla calibrazione dell’incertezza in presenza
di distribution shift. Il sistema principale combina moduli di convoluzione di tipo GCN e
GIN con blocchi temporali TCN per catturare congiuntamente le dipendenze topologiche e le
dinamiche temporali del flusso di traffico. A partire da solide baseline di persistenza e da modelli
puramente temporali senza grafo, sono state esplorate e confrontate varianti architetturali
significative, analizzando differenti meccanismi di propagazione spaziale, formulazioni della
testa probabilistica (regressione quantilica via pinball loss) e variazioni nella connettività del
grafo. La valutazione, condotta cronologicamente senza leakage sui dataset di riferimento
(METR-LA) per orizzonti di 3, 6 e 12 step, evidenzia le prestazioni sia in termini di accuratezza
puntuale (MAE, RMSE) sia di affidabilità degli intervalli predittivi. Infine, uno stress test
controllato sotto distribution shift dimostra come la componente su grafo influisca non solo
sull’errore di previsione, ma anche sulla capacità del modello di quantificare correttamente la
propria confidenza in condizioni anomale.

Link al documento: https://www.overleaf.com/project/6ab6b9a459e0c3d149e94cbd/share#44bcb46530411940343bac4e0ca4902ff0f07ad769edf83e
