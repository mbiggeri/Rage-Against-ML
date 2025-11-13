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

- Implementare altri modelli interessanti

- Implementare ricerca con Optuna
-----

## Esempi di script per avviare il programma

python main.py --model standard --dataset monk1 --activation tanh --hidden_sizes 3 --epochs 1000 --batch_size 64 --lr 0.001

python main.py --model standard --dataset ml_cup --activation tanh --hidden_sizes 50 20 --epochs 150 --batch_size 64 --lr 0.001

python main.py --model standard --dataset mnist --activation sigmoid --hidden_sizes 256 128 64 --epochs 10 --batch_size 64

python main.py --model step_out --dataset mnist --activation sigmoid --hidden_sizes 256 128 64 --epochs 10 --batch_size 64