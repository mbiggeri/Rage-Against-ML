# RAGE AGAINST MACHINE LEARNING

1) All'inizio del main trovate tutti i parametri con cui chiamare i modelli per definire la rete, dataset, ecc... e far partire l'addestramento;

2) Per definire nuovi modelli fate un file apposito in models e poi implementate la chiamata nel main così manteniamo tutto ordinato;

3) Non ho aggiunto niente delle SVM ho solo messo l'import della libreria all'inizio;

4) Appena lo trovo aggiungo al progetto anche il file che avevo nella tesi per la ricerca degli iperparametri, così si può adattare al nostro main (sarà un file simile perché deve inizializzare tutto allo stesso modo, con la differenza che dovremo definire della roba in più per impostare i parametri di ricerca di optuna)

5) un esempio di riga di comando per far partire un modello standard (MLP) per 1000 epoche con il monc è:
python main.py --model standard --dataset monk1 --epochs 1000 --hidden_sizes 3 3

6) Sarebbe utile implementare un early-stopping per evitare che il modello overfitti senza dover cercare un numero preciso di epoche di addestramento

7) Vi aggiungo poi anche altra roba che era utilizzata nella tesi per testare diverse funzioni di attivazione, aggiungere il momentum come iperparametro, ecc...