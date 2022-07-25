# TV-Show-Recommender
NLU-based recommendation system for TV shows

We scrape descriptions of TV shows from the IMDb website, and use the pre-trained DistilBert model to distill the essence of each TV show into a single vector.

Then, a user-entered sentence is transformed into another vector using the same DistilBert model. 
We compare this to every other TV show (using cosine similarity), and then return the top 5 most similar shows.

Last Updated: 07/25/2022
