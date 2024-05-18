
# YouTube Data Science Analysis

This project analyzes YouTube video data, specifically focusing on the 3Blue1Brown channel. The analysis includes data retrieval from the YouTube API, sentiment analysis of comments, and topic modeling of video transcripts. The aim is to identify key factors contributing to video popularity and to explore patterns in viewer engagement.

## Table of Contents
- [Installation](#installation)
- [Data Collection](#data-collection)
- [Data Analysis](#data-analysis)
- [Visualizations](#visualizations)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, clone this repository and install the required packages. 

```bash
git clone https://github.com/ahnaft80/3blue1brown_analysis
cd 3blue1brown_analysis
```

Ensure you have the following libraries installed:
- `youtube-dl`
- `pandas`
- `matplotlib`
- `seaborn`
- `youtube-transcript-api`
- `google-api-python-client`
- `isodate`
- `scikit-learn`
- `transformers`
- `wordcloud`

## Data Collection

The data is collected using the YouTube API. The process involves:
1. Fetching video details including publication date and duration.
2. Retrieving comments for each video.
3. Extracting transcripts of the videos.

API keys and client setup are required to access the YouTube API. Ensure you have a valid API key and update the key in the notebook.

## Data Analysis

### Sentiment Analysis

Sentiment analysis is performed on the comments using the VADER sentiment analysis tool. The sentiment scores are calculated and analyzed to understand viewer reactions to the videos.

### Topic Modeling

Latent Dirichlet Allocation (LDA) is used to model topics from video transcripts. This helps in identifying recurring themes and concepts in the videos.

### Correlation Analysis

Correlation analysis is performed to identify relationships between various metrics such as sentiment scores, video popularity, and topic words.

## Visualizations

Visualizations include:
- Scatter plots showing the correlation between sentiment scores and video popularity.
- Word clouds displaying the most common topic words in top videos.
- Bar charts showing average popularity scores for different clusters of topics.

## Results

The analysis reveals insights into the factors contributing to video popularity, such as:
- Viewer sentiment and its impact on engagement.
- Recurring topics in popular videos.
- Patterns in comment sentiment and their relation to video metrics.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
