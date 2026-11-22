# Error analysis on the shared test subset

Counts out of the same utterances for every run. Slot errors are counted only where the intent is right, so they are not double-counted with intent errors.

## Intent and format errors

| Run | Format failures | Intent wrong | of which invented label | Intent right, slots wrong |
|---|---:|---:|---:|---:|
| qwen05b-zero | 40 | 732 | 306 | 259 |
| qwen05b-few | 3 | 226 | 4 | 295 |
| qwen05b-lora | 0 | 169 | 0 | 225 |
| qwen7b-zero | 29 | 326 | 14 | 448 |
| qwen7b-few | 0 | 125 | 3 | 237 |

## Slot errors where the intent is right

| Run | boundary | wrong type | missed | extra | wrong value |
|---|---:|---:|---:|---:|---:|
| qwen05b-zero | 21 | 74 | 108 | 382 | 20 |
| qwen05b-few | 52 | 22 | 203 | 107 | 12 |
| qwen05b-lora | 71 | 28 | 101 | 96 | 4 |
| qwen7b-zero | 63 | 109 | 200 | 319 | 5 |
| qwen7b-few | 70 | 33 | 99 | 118 | 1 |

Boundary = right type, overlapping value; wrong type = right value, other type; wrong value = right type, unrelated value.

## Slot F1 by type (most frequent gold types)

| Slot type | Gold slots | qwen05b-zero | qwen05b-few | qwen05b-lora | qwen7b-zero | qwen7b-few |
|---|---:|---:|---:|---:|---:|---:|
| date | 132 | 21.3 | 57.4 | 80.6 | 20.7 | 79.0 |
| place_name | 94 | 11.7 | 64.5 | 78.7 | 46.9 | 82.2 |
| event_name | 84 | 1.8 | 53.3 | 66.3 | 18.5 | 62.5 |
| time | 75 | 6.1 | 59.1 | 70.3 | 39.1 | 69.6 |
| person | 75 | 22.1 | 60.7 | 73.9 | 52.9 | 74.5 |
| media_type | 43 | 0.0 | 70.0 | 77.6 | 4.9 | 75.3 |
| house_place | 30 | 0.0 | 70.6 | 94.7 | 22.9 | 84.2 |
| business_name | 29 | 0.0 | 72.0 | 56.7 | 33.3 | 82.1 |
| list_name | 28 | 6.7 | 34.6 | 72.0 | 32.8 | 65.2 |
| weather_descriptor | 27 | 0.0 | 65.3 | 67.8 | 36.1 | 72.7 |
| transport_type | 25 | 13.3 | 82.8 | 96.0 | 62.7 | 93.9 |
| food_type | 24 | 12.9 | 66.7 | 74.6 | 0.0 | 73.5 |

## Most frequent intent confusions: `qwen05b-zero`

| Gold | Predicted | Count |
|---|---|---:|
| play_music | play_audiobook | 23 |
| qa_factoid | general_quirky | 16 |
| iot_hue_lightoff | general_quirky | 14 |
| general_quirky | general_greet | 11 |
| qa_definition | general_quirky | 11 |
| music_likeness | recommendation_movies | 10 |
| play_music | music_query | 8 |
| general_quirky | news_query | 8 |
| transport_ticket | transport_query | 8 |
| calendar_query | datetime_query | 7 |

## Most frequent intent confusions: `qwen05b-few`

| Gold | Predicted | Count |
|---|---|---:|
| general_quirky | qa_definition | 5 |
| calendar_set | lists_createoradd | 4 |
| calendar_set | alarm_set | 4 |
| qa_factoid | general_quirky | 4 |
| qa_factoid | qa_definition | 4 |
| general_quirky | qa_factoid | 3 |
| general_quirky | general_greet | 3 |
| calendar_set | datetime_query | 3 |
| general_quirky | lists_createoradd | 3 |
| calendar_query | lists_query | 3 |

## Most frequent intent confusions: `qwen05b-lora`

| Gold | Predicted | Count |
|---|---|---:|
| general_quirky | qa_factoid | 8 |
| calendar_set | calendar_query | 7 |
| calendar_query | calendar_set | 6 |
| qa_definition | general_quirky | 4 |
| play_music | play_radio | 3 |
| general_quirky | calendar_query | 3 |
| calendar_query | recommendation_events | 3 |
| play_game | play_music | 3 |
| qa_factoid | general_quirky | 3 |
| news_query | calendar_query | 2 |

## Most frequent intent confusions: `qwen7b-zero`

| Gold | Predicted | Count |
|---|---|---:|
| cooking_recipe | cooking_query | 22 |
| datetime_query | datetime_convert | 18 |
| calendar_set | alarm_set | 12 |
| calendar_set | alarm_query | 11 |
| general_quirky | qa_factoid | 8 |
| calendar_set | calendar_query | 8 |
| qa_factoid | weather_query | 8 |
| weather_query | transport_traffic | 6 |
| general_quirky | social_query | 6 |
| calendar_query | alarm_query | 5 |

## Most frequent intent confusions: `qwen7b-few`

| Gold | Predicted | Count |
|---|---|---:|
| general_quirky | qa_factoid | 9 |
| calendar_set | calendar_query | 7 |
| cooking_recipe | cooking_query | 4 |
| general_quirky | qa_definition | 3 |
| news_query | qa_factoid | 2 |
| news_query | recommendation_events | 2 |
| general_quirky | calendar_query | 2 |
| general_quirky | news_query | 2 |
| general_quirky | recommendation_locations | 2 |
| calendar_query | recommendation_events | 2 |

## Where `qwen05b-lora` and `qwen7b-few` fail (exact match)

| Both right | Only qwen05b-lora wrong | Only qwen7b-few wrong | Both wrong | Both give the same wrong answer |
|---:|---:|---:|---:|---:|
| 497 | 141 | 109 | 253 | 71 |

Two different models agreeing on the same non-gold answer is the first place to look for label noise or ambiguous requests:

| Utterance | Gold | Both models |
|---|---|---|
| what is the exchange rate of us dollar to pound sterling | `news_query [["news_topic", "exchange rate of us dollar to pound sterling"]]` | `qa_currency [["currency_name", "us dollar"], ["currency_name", "pound sterling"]]` |
| mild or heavy weather | `weather_query [["weather_descriptor", "mild"], ["weather_descriptor", "heavy"]]` | `weather_query [["weather_descriptor", "mild or heavy"]]` |
| robot vacuum the living room now | `iot_cleaning [["house_place", "living room"]]` | `iot_cleaning [["device_type", "robot vacuum"], ["house_place", "living room"]]` |
| play some christian music | `play_music [["music_descriptor", "christian"]]` | `play_music [["music_genre", "christian"]]` |
| what day of the week does the fifteenth of march fall on | `datetime_query [["date", "fifteenth of march"]]` | `datetime_query [["date", "the fifteenth of march"]]` |
| fast | `audio_volume_up []` | `general_quirky []` |
| light color for study room | `iot_hue_lightup [["house_place", "study room"]]` | `iot_hue_lightchange [["house_place", "study room"]]` |
| seven hundred | `datetime_convert []` | `qa_maths []` |
| daily news | `news_query []` | `news_query [["general_frequency", "daily"]]` |
| are there any forecasts of rain or snow this week | `weather_query [["weather_descriptor", "rain or snow"], ["date", "this week"]]` | `weather_query [["weather_descriptor", "rain"], ["weather_descriptor", "snow"], ["date", "this week"]]` |
| is the twenty second on a wednesday | `datetime_query [["date", "twenty second"]]` | `datetime_query [["date", "twenty second"], ["date", "wednesday"]]` |
| change the lights in the living room to green and red | `iot_hue_lightchange [["house_place", "living room"], ["color_type", "green and red"]]` | `iot_hue_lightchange [["house_place", "living room"], ["color_type", "green"], ["color_type", "red"]]` |
| play my midnight love playlist | `play_music [["playlist_name", "midnight love"]]` | `play_music [["playlist_name", "midnight love playlist"]]` |
| i'd like to listen to hear a music of dance and country | `play_music [["music_genre", "dance and country"]]` | `play_music [["music_genre", "dance"], ["music_genre", "country"]]` |
| tell me when it is five p. m. | `alarm_set [["time", "five p. m."]]` | `datetime_query [["time", "five p. m."]]` |
| lower the overhead lights | `iot_hue_lightdim [["house_place", "overhead"]]` | `iot_hue_lightdim []` |
| would you love to see | `iot_hue_lightchange []` | `general_quirky []` |
| s. | `play_music []` | `general_quirky []` |
| what's on the radio right now | `music_query []` | `play_radio []` |
| how old are you | `general_quirky []` | `qa_factoid []` |

Produced by `ftvp-errors` from `reports/runs/*.jsonl`.
