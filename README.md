# TV-Show-Recommender
NLU-based recommendation system for TV shows

We scrape descriptions of TV shows from the IMDb website, and use the pre-trained Bert models to distill the essence of each TV show into a single vector.

Then, a user-entered sentence is transformed into another vector using the same Bert model. We compare this to every other TV show (using cosine similarity), and then return the top 10 most similar shows.
