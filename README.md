# TV-Show-Recommender
NLU-based recommendation system for TV shows

We scrape descriptions of TV shows from the IMDb website, and use the pre-trained DistilBert model to distill the essence of each TV show into a single vector.

Then, a user-entered sentence is transformed into another vector using the same DistilBert model. 
We compare this to every other TV show (using cosine similarity), and then return the top 5 most similar shows.

Update 07/31/2022:
You can check out the Flask app at https://niccolo-recommender-app.herokuapp.com/. Please allow for a minute or so for the container to spin up.

Last Updated: 07/31/2022
