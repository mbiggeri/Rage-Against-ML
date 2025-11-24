# RAGE AGAINST MACHINE LEARNING

1) All'inizio del main trovate tutti i parametri con cui chiamare i modelli per definire la rete, dataset, ecc... e far partire l'addestramento;

2) Per definire nuovi modelli fate un file apposito in models e poi implementate la chiamata nel main così manteniamo tutto ordinato;

3) Non ho aggiunto niente delle SVM ho solo messo l'import della libreria all'inizio;

4) Appena lo trovo aggiungo al progetto anche il file che avevo nella tesi per la ricerca degli iperparametri, così si può adattare al nostro main (sarà un file simile perché deve inizializzare tutto allo stesso modo, con la differenza che dovremo definire della roba in più per impostare i parametri di ricerca di optuna)

5) Sarebbe utile implementare un early-stopping per evitare che il modello overfitti senza dover cercare un numero preciso di epoche di addestramento
-----

## Cose da fare:

- Implementare Grid search (dalle slide sembra obbligatorio farlo prima di provate altri metodi, come Optuna)

- Implementare SVM

- Per MLP: 
1. implementare dropout e L2 regularization
2. Controllare come influisce lo scaling dei dati in input

- Implementare altri modelli interessanti

- Provare a implementare Random Forest
-----

## Esempi di script per avviare il programma

python main.py --model standard --dataset monk1 --activation tanh --hidden_sizes 3 --epochs 1000 --batch_size 64 --lr 0.001

python main.py --model standard --dataset ml_cup --activation tanh --hidden_sizes 50 20 --epochs 150 --batch_size 64 --lr 0.001

python main.py --model standard --dataset mnist --activation sigmoid --hidden_sizes 256 128 64 --epochs 10 --batch_size 64

python main.py --model step_out --dataset mnist --activation sigmoid --hidden_sizes 256 128 64 --epochs 10 --batch_size 64
-----

## Ricerca Iperparametri (Optuna)

Lo script `optuna_search.py` permette di cercare automaticamente la migliore combinazione di parametri (learning rate, numero di neuroni, funzioni di attivazione e tutto quello che vuoi amore).
Puoi cambiare gli insiemi da cui Optuna può pescare gli iperparametri in questo modo:
    -> *batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])* diventa *batch_size = trial.suggest_categorical("batch_size", [64, 128])* per escludere il valore 32 dalla scelta (suggest_categorical sceglie uno tra i valori indicati).
    -> *n_layers = trial.suggest_int("n_layers", 1, 3)* diventa *n_layers = trial.suggest_int("n_layers", 4, 6)* per cercare solo reti con un numero compreso tra 4 e 6 hidden layers (suggest_int sceglie un valore intero compreso tra il primo e il secondo indicati)
    -> Penso che il concetto sia chiaro anche senza fare altri esempi di suggest_xyz.

### Come funziona
Lo script definisce una "funzione obiettivo" che restituisce un punteggio per ogni configurazione testata. Optuna cercherà automaticamente di massimizzare questo punteggio se è una metrica di bontà (Accuratezza per MONK/MNIST/...) o di minimizzarlo se è un errore (MSE per ML-CUP). In ogni esecuzione (Trial):
1.  Optuna testa dei parametri "promettenti"*.
2.  Viene addestrata una rete con quei parametri.
3.  Se la rete sta andando male rispetto alle altre, Optuna la interrompe subito (**Pruning**) per risparmiare tempo. Vengono fatti in sostanza meno trial di quelli indicati, perché alcuni saranno pessimi.
4.  Alla fine, salva i parametri migliori in `results/best_params_<dataset>.json`.


* A differenza della ricerca casuale (Random Search) o della griglia (Grid Search), Optuna utilizza di default un approccio Bayesiano chiamato Tree-structured Parzen Estimator (TPE).

In parole semplici, Optuna costruisce un modello probabilistico basato sulla storia dei trial passati. Divide le configurazioni provate in "buone" e "cattive" e cerca di capire quali valori degli iperparametri (es. un learning rate basso o alto) sono correlati ai risultati migliori. Al trial successivo, non "tira a indovinare", ma pesca i parametri dalle zone che il modello ritiene più promettenti, concentrando lo sforzo dove è più probabile trovare l'ottimo.

Il paper: [Optuna: A Next-generation Hyperparameter Optimization Framework] https://arxiv.org/abs/1907.10902

### Esempio di utilizzo
```bash
# Per classificazione (MONK-1), esegue 100 tentativi ognuno per 20 epoche.
python optuna_search.py --dataset monk1 --trials 100 --epochs 20

# Per regressione (ML-CUP), esegue 50 tentativi ognuno con 50 epoche.
python optuna_search.py --dataset mlc25 --trials 50 --epochs 50