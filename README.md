# Gen-Wenyan
La littérature chinoise ancienne se divise principalement en Prose 文 (ex: annales historiques comme le Shiji) et en Poésie 詩 (ex: Poésie Tang). Ces deux formes s'opposent fondamentalement : la prose se caractérise par une narration fluide et des phrases de longueurs variables privilégiant la cohérence sémantique, tandis que la poésie respecte des contraintes strictes de rythme et de rimes. Un modèle unique entraîné sur des données mixtes peine souvent à concilier ces objectifs contradictoires, risquant de produire une prose rigide ou une poésie sans rythme.

Pour traiter ce problème, nous proposons un système de génération à double flux sensible au style. Au lieu d'entraîner un modèle monolithique, nous séparons la tâche en deux sous-tâches spécialisées. 

Notre système comprend donc deux composants:

- Un Routeur/Classificateur qui identifie le style de l'entrée.
- Deux Modèles Génératifs, l'un dédié à la Prose Historique et l'autre à la Poésie Classique.

## Structure du projet

Le projet est organisé comme suit :

- **`src/`** : Contient l'ensemble du code source.
  - **`scripts_prose/train/`** : Scripts d'entraînement pour le modèle de **Prose** et le **Classificateur** de style.
  - **`scripts_poem/script_train/`** : Scripts d'entraînement dédiés à la **Poésie**.
  - **`inference/`** : Pipeline d'exécution (`pipeline.py`) et logique de génération de texte (`generate.py`).
  - **`analyze_data.py`** : Script pour l'analyse statistique des corpus (longueur, distribution).
- **`data/`** : Répertoire de stockage des données.
  - **`processed/`** : Données nettoyées, tenseurs d'entraînement (`.pt`) et dictionnaire (`vocab.json`).

