# Crawling branch

## Global meme directory structure

- `meme_characters`
   * `character_1`
      + `character_1.csv`
	  + `character_1_metadata.csv`
	  + `character_1.jpg`
   * `character_2`
	  + `character_2.csv`
	  + `character_2_metadata.csv`
	  + `character_2.jpg`
   *     .
   *     .
   *     .
   * `character_n`
      + `character_n.csv`
	  + `character_n_metadata.csv`
	  + `character_n.jpg`

This structure is usually located at `meme_crawler/meme_crawler/`

## LSTM text generation script usage

```bash
cd meme_crawler/scripts
```
(Change the `global_dir` variable (line 31 of [`lstm_text_generation.py`](https://github.com/alorozco53/Deep-Meme-Captioner/blob/crawling/meme_crawler/scripts/lstm_text_generation.py)) to directory pointing to global structure as defined previously.)

```bash
python lstm_text_generation.py
```

## Getting stats for the available memes

```bash
cd meme_crawler/scripts
python meme_stats.py <path-to-global-meme-dir>
```

## Running the crawler

```bash
cd meme_crawler/
python scrapy crawl memecaptionspider -s JOBDIR=<previous-state-dir>
```

where `<previous-state-dir>` is the path to the directory where the spider's last state is stored.
