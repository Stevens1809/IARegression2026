from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model=pickle.load(open('modelRL.pkl', 'rb')) # chargement du modèle à partir du fichier 'model.pkl'

@app.route('/')
def home(): # route pour la page d'accueil, affiche le formulaire de saisie
    return render_template('index.html')

@app.route('/predict', methods=['POST']) # route pour la prédiction, accepte uniquement les requêtes POST
def predict(): # fonction pour faire la prédiction à partir des données du formulaire
    features = [ float(x) for x in request.form.values() ] # liste de comprehension pour convertir les valeurs du formulaire en entiers
    import numpy as np
    features_finals= [np.array(features)] # conversion de la liste de features en un tableau numpy
    prediction = model.predict(features_finals) # utilisation du modèle pour faire une prédiction
    output = round(prediction[0], 2) # arrondi de la prédiction à 2 décimales
    return render_template('index.html', prediction_text=f"La charge d'assurance est de {output}") # affichage du résultat dans la page HTML
if __name__ == "__main__":
    app.run(debug=True) # lancement de l'application Flask en mode debug
