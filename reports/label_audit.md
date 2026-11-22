# Manual audit: same wrong answer from both strong systems

The 71 test utterances where the LoRA model (`qwen05b-lora`) and the 7B few-shot model (`qwen7b-few`) return the identical answer and it differs from the gold label (from `ftvp-errors --pair qwen05b-lora qwen7b-few`). Each was read once by one person (me) and put in one of three buckets. This is a judgement, not a second annotation pass.

| Bucket | Count |
|---|---:|
| gold looks wrong / unlabelable | 13 |
| defensible, other convention | 43 |
| model wrong | 15 |

| Id | Utterance | Gold | Both models | Bucket |
|---|---|---|---|---|
| 281 | what is the exchange rate of us dollar to pound sterling | `news_query [["news_topic", "exchange rate of us dollar to pound sterling"]]` | `qa_currency [["currency_name", "us dollar"], ["currency_name", "pound sterling"]]` | gold looks wrong / unlabelable |
| 790 | mild or heavy weather | `weather_query [["weather_descriptor", "mild"], ["weather_descriptor", "heavy"]]` | `weather_query [["weather_descriptor", "mild or heavy"]]` | defensible, other convention |
| 864 | robot vacuum the living room now | `iot_cleaning [["house_place", "living room"]]` | `iot_cleaning [["device_type", "robot vacuum"], ["house_place", "living room"]]` | defensible, other convention |
| 891 | play some christian music | `play_music [["music_descriptor", "christian"]]` | `play_music [["music_genre", "christian"]]` | defensible, other convention |
| 1098 | what day of the week does the fifteenth of march fall on | `datetime_query [["date", "fifteenth of march"]]` | `datetime_query [["date", "the fifteenth of march"]]` | defensible, other convention |
| 1300 | fast | `audio_volume_up []` | `general_quirky []` | gold looks wrong / unlabelable |
| 2264 | light color for study room | `iot_hue_lightup [["house_place", "study room"]]` | `iot_hue_lightchange [["house_place", "study room"]]` | defensible, other convention |
| 2451 | seven hundred | `datetime_convert []` | `qa_maths []` | gold looks wrong / unlabelable |
| 2496 | daily news | `news_query []` | `news_query [["general_frequency", "daily"]]` | defensible, other convention |
| 3407 | are there any forecasts of rain or snow this week | `weather_query [["weather_descriptor", "rain or snow"], ["date", "this week"]]` | `weather_query [["weather_descriptor", "rain"], ["weather_descriptor", "snow"], ["date", "this week"]]` | defensible, other convention |
| 3441 | is the twenty second on a wednesday | `datetime_query [["date", "twenty second"]]` | `datetime_query [["date", "twenty second"], ["date", "wednesday"]]` | defensible, other convention |
| 3542 | change the lights in the living room to green and red | `iot_hue_lightchange [["house_place", "living room"], ["color_type", "green and red"]]` | `iot_hue_lightchange [["house_place", "living room"], ["color_type", "green"], ["color_type", "red"]]` | defensible, other convention |
| 3780 | play my midnight love playlist | `play_music [["playlist_name", "midnight love"]]` | `play_music [["playlist_name", "midnight love playlist"]]` | defensible, other convention |
| 3912 | i'd like to listen to hear a music of dance and country | `play_music [["music_genre", "dance and country"]]` | `play_music [["music_genre", "dance"], ["music_genre", "country"]]` | defensible, other convention |
| 4601 | tell me when it is five p. m. | `alarm_set [["time", "five p. m."]]` | `datetime_query [["time", "five p. m."]]` | model wrong |
| 4896 | lower the overhead lights | `iot_hue_lightdim [["house_place", "overhead"]]` | `iot_hue_lightdim []` | model wrong |
| 4955 | would you love to see | `iot_hue_lightchange []` | `general_quirky []` | gold looks wrong / unlabelable |
| 5378 | s. | `play_music []` | `general_quirky []` | gold looks wrong / unlabelable |
| 5614 | what's on the radio right now | `music_query []` | `play_radio []` | defensible, other convention |
| 6248 | how old are you | `general_quirky []` | `qa_factoid []` | defensible, other convention |
| 6366 | tell me some best tourist places to visit in america | `recommendation_locations [["place_name", "best tourist places"], ["place_name", "america"]]` | `recommendation_locations [["place_name", "america"]]` | model wrong |
| 6424 | hello how is your day | `general_quirky []` | `general_greet []` | defensible, other convention |
| 6980 | give me to do list | `calendar_query []` | `lists_query []` | defensible, other convention |
| 7057 | schedule a lunch meeting with brent on wednesday at noon | `calendar_set [["event_name", "lunch meeting"], ["person", "brent"], ["date", "wednesday"], ["timeofday", "noon"]]` | `calendar_set [["meal_type", "lunch"], ["person", "brent"], ["date", "wednesday"], ["timeofday", "noon"]]` | defensible, other convention |
| 7353 | just let me know next weeks weather in california | `email_query [["time", "next weeks"], ["place_name", "california"]]` | `weather_query [["date", "next weeks"], ["place_name", "california"]]` | gold looks wrong / unlabelable |
| 7559 | remind me tonight to pick up my dry cleaning at eight p. m. | `calendar_set [["event_name", "pick up my dry cleaning"], ["time", "eight p. m."]]` | `calendar_set [["timeofday", "tonight"], ["event_name", "pick up my dry cleaning"], ["time", "eight p. m."]]` | defensible, other convention |
| 7606 | meet with joe tomorrow at three | `calendar_set [["person", "joe"], ["date", "tomorrow"], ["time", "three"]]` | `calendar_set [["event_name", "meet"], ["person", "joe"], ["date", "tomorrow"], ["time", "three"]]` | defensible, other convention |
| 7946 | what are the upcoming concerts in dallas | `calendar_query [["place_name", "dallas"]]` | `recommendation_events [["event_name", "concerts"], ["place_name", "dallas"]]` | defensible, other convention |
| 7980 | tell me when i have an event | `calendar_set []` | `calendar_query []` | gold looks wrong / unlabelable |
| 8345 | let me know when the meeting is gonna happen | `calendar_set [["event_name", "meeting"]]` | `calendar_query [["event_name", "meeting"]]` | gold looks wrong / unlabelable |
| 8388 | i want to do laundry at eight p. m. create reminder | `calendar_set [["time", "eight p. m."]]` | `calendar_set [["event_name", "laundry"], ["time", "eight p. m."]]` | defensible, other convention |
| 9043 | upcoming events in calendar | `calendar_set [["event_name", "upcoming events"]]` | `calendar_query []` | gold looks wrong / unlabelable |
| 9406 | start sirius xm radio channel | `play_radio [["radio_name", "sirius xm radio channel"]]` | `play_radio [["app_name", "sirius xm"]]` | defensible, other convention |
| 9601 | play country radio | `play_radio [["radio_name", "country radio"]]` | `play_radio [["radio_name", "country"]]` | defensible, other convention |
| 9619 | play prairie home companion on the car radio | `play_radio [["device_type", "car radio"]]` | `play_radio [["radio_name", "prairie home companion"]]` | model wrong |
| 9895 | keep playing secret garden | `play_audiobook [["player_setting", "keep playing"], ["audiobook_name", "secret garden"]]` | `play_music [["song_name", "secret garden"]]` | model wrong |
| 9931 | olly how long should i boil the egg | `cooking_recipe [["cooking_type", "boil"], ["food_type", "egg"]]` | `cooking_recipe [["food_type", "egg"]]` | defensible, other convention |
| 9941 | how do i cook butter chicken | `cooking_recipe [["cooking_type", "cook"], ["food_type", "butter chicken"]]` | `cooking_recipe [["food_type", "butter chicken"]]` | defensible, other convention |
| 9956 | what can i substitute butter with | `cooking_recipe [["ingredient", "butter"]]` | `cooking_recipe [["food_type", "butter"]]` | model wrong |
| 9971 | what conducts heat better copper bottomed pots or cast iron | `cooking_recipe []` | `qa_factoid []` | defensible, other convention |
| 10214 | recipes that can be cooked in an hour | `cooking_recipe []` | `cooking_recipe [["time", "an hour"]]` | defensible, other convention |
| 10525 | remove my car payment on my calendar | `lists_remove []` | `calendar_remove [["event_name", "car payment"]]` | defensible, other convention |
| 10579 | reset my locations list | `lists_remove [["list_name", "locations"]]` | `lists_remove []` | model wrong |
| 10645 | add red wine to my shopping list | `lists_createoradd [["list_name", "shopping list"]]` | `lists_createoradd [["list_name", "shopping"]]` | defensible, other convention |
| 10667 | recite the schedules of the list | `lists_query [["list_name", "schedules"]]` | `lists_query []` | model wrong |
| 10687 | if there is event of show then remove it | `lists_remove []` | `calendar_remove []` | defensible, other convention |
| 10936 | please remove my list of favorite albums | `lists_remove [["list_name", "favorite albums"]]` | `lists_remove []` | model wrong |
| 10996 | add bob to my list of contacts | `lists_createoradd [["person", "bob"], ["list_name", "list of contacts"]]` | `email_addcontact [["person", "bob"]]` | model wrong |
| 11107 | what are my current list | `lists_query [["list_name", "current"]]` | `lists_query []` | model wrong |
| 11177 | add a wrist watch to the shopping list | `lists_createoradd [["list_name", "shopping list"]]` | `lists_createoradd [["list_name", "shopping"]]` | defensible, other convention |
| 11259 | jump to next podcast | `play_podcasts [["player_setting", "jump"], ["podcast_descriptor", "next podcast"]]` | `play_podcasts []` | model wrong |
| 11306 | play that podcast i was listening to yesterday | `play_podcasts [["podcast_descriptor", "listening to yesterday"]]` | `play_podcasts [["date", "yesterday"]]` | defensible, other convention |
| 11529 | hey play my favorite podcast from list | `play_podcasts [["podcast_descriptor", "favorite"]]` | `play_podcasts []` | model wrong |
| 12606 | i need to get a ticket via train to orlando from hwood | `transport_ticket [["transport_type", "train"], ["place_name", "orlando from hwood"]]` | `transport_ticket [["transport_type", "train"], ["place_name", "orlando"], ["place_name", "hwood"]]` | defensible, other convention |
| 13077 | what is robin williams birthday | `qa_factoid [["person", "robin williams"]]` | `qa_factoid [["person", "robin williams"], ["event_name", "birthday"]]` | defensible, other convention |
| 13083 | drive the car and bring my friends over | `general_quirky []` | `transport_taxi [["transport_type", "car"], ["relation", "friends"]]` | defensible, other convention |
| 13175 | check celebrity where abouts | `general_quirky []` | `qa_factoid []` | defensible, other convention |
| 13242 | what is the best chocolate chip cookies recipe | `general_quirky []` | `cooking_recipe [["food_type", "chocolate chip cookies"]]` | defensible, other convention |
| 13486 | can you show me a list of all my recent commands | `general_quirky []` | `lists_query []` | defensible, other convention |
| 13982 | dollar rate at present | `qa_currency [["currency_name", "dollar"]]` | `qa_currency []` | defensible, other convention |
| 14038 | what is the location of moldova | `qa_factoid []` | `qa_factoid [["place_name", "moldova"]]` | defensible, other convention |
| 14457 | find route | `qa_factoid []` | `transport_query []` | gold looks wrong / unlabelable |
| 15138 | check my social networks | `social_query [["media_type", "social networks"]]` | `social_query []` | model wrong |
| 15187 | share in friends post | `social_post []` | `social_post [["relation", "friends"]]` | defensible, other convention |
| 15225 | show me latest notifications | `social_query []` | `calendar_query []` | defensible, other convention |
| 15305 | do i have any notifications from social media | `calendar_query []` | `social_query []` | gold looks wrong / unlabelable |
| 16032 | please send an email to sally regarding tomorrows meeting | `email_sendemail [["person", "sally"], ["date", "tomorrows"], ["event_name", "meeting"]]` | `email_sendemail [["person", "sally"], ["event_name", "tomorrows meeting"]]` | defensible, other convention |
| 16168 | have any emails arrived in the last fifteen minutes | `email_query [["time", "fifteen minutes"]]` | `email_query [["time", "last fifteen minutes"]]` | defensible, other convention |
| 16518 | add a new email for julie smith | `email_sendemail [["person", "julie smith"]]` | `email_addcontact [["person", "julie smith"]]` | gold looks wrong / unlabelable |
| 16519 | i need to add a new email to my contacts | `email_sendemail []` | `email_addcontact []` | gold looks wrong / unlabelable |
| 17111 | contact detail of paul | `email_querycontact [["personal_info", "contact detail"], ["person", "paul"]]` | `email_querycontact [["person", "paul"]]` | model wrong |
