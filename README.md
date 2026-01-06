# WenyanGen
La génération de texte en Chinois Classique (Wenyan) pose des défis uniques aux modèles de langage actuels (LLMs), notamment en raison de la rareté des corpus de qualité et des différences stylistiques extrêmes entre les genres.

La littérature chinoise ancienne se divise principalement en Prose 文 (ex: annales historiques comme le Shiji) et en Poésie 詩 (ex: Poésie Tang). Ces deux formes s'opposent fondamentalement : la prose se caractérise par une narration fluide et des phrases de longueurs variables privilégiant la cohérence sémantique, tandis que la poésie respecte des contraintes strictes de rythme et de rimes. Un modèle unique entraîné sur des données mixtes peine souvent à concilier ces objectifs contradictoires, risquant de produire une prose rigide ou une poésie sans rythme.

Pour traiter ce problème, nous proposons un système de génération à double flux sensible au style. Au lieu d'entraîner un modèle monolithique, nous séparons la tâche en deux sous-tâches spécialisées. 

Notre système comprend donc deux composants:

- Un Routeur/Classificateur qui identifie le style de l'entrée.
- Deux Modèles Génératifs de type Transformer, l'un dédié à la Prose Historique et l'autre à la Poésie Classique.

Cette approche permet à chaque modèle de capturer les nuances de son genre spécifique. Dans ce projet, nous implémentons cette architecture à l'aide d'un Transformer de type GPT personnalisé.
