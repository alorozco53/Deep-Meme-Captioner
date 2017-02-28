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

## Getting stats for the available memes

```bash
cd meme_crawler/scripts
python meme_stats.py <path-to-global-meme-dir>
```

## Running the crawler

```bash
cd meme_crawler/
scrapy crawl memecaptionspider -s JOBDIR=<previous-state-dir>
```

where `<previous-state-dir>` is the path to the directory where the spider's last state is stored.
