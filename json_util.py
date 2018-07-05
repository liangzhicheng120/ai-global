#!/bin/bash
# -*-coding=utf-8-*-
import json
from util.file_util import *
from util.log_util import *
import random

name = '80'
fileUtil = FileUtil()
files, _ = fileUtil.get_files('E:\\LearningDeepBatch\\{0}\\'.format(name))
result = []
print(len(files))
for image_id in files:
    item = {}
    item['image_id'] = image_id
    if 'airport_terminal' in image_id:
        item['label_id'] = '0'
    if 'landing_field' in image_id:
        item['label_id'] = '1'
    if 'airplane_cabin' in image_id:
        item['label_id'] = '2'
    if 'amusement_park' in image_id:
        item['label_id'] = '3'
    if 'skating_rink' in image_id:
        item['label_id'] = '4'
    if 'arena_performance' in image_id:
        item['label_id'] = '5'
    if 'art_room' in image_id:
        item['label_id'] = '6'
    if 'assembly_line' in image_id:
        item['label_id'] = '7'
    if 'baseball_field' in image_id:
        item['label_id'] = '8'
    if 'football_field' in image_id:
        item['label_id'] = '9'
    if 'soccer_field' in image_id:
        item['label_id'] = '10'
    if 'volleyball_court' in image_id:
        item['label_id'] = '11'
    if 'golf_course' in image_id:
        item['label_id'] = '12'
    if 'athletic_field' in image_id:
        item['label_id'] = '13'
    if 'ski_slope' in image_id:
        item['label_id'] = '14'
    if 'basketball_court' in image_id:
        item['label_id'] = '15'
    if 'gymnasium' in image_id:
        item['label_id'] = '16'
    if 'bowling_alley' in image_id:
        item['label_id'] = '17'
    if 'swimming_pool' in image_id:
        item['label_id'] = '18'
    if 'boxing_ring' in image_id:
        item['label_id'] = '19'
    if 'racecourse' in image_id:
        item['label_id'] = '20'
    if 'farm_farm_field' in image_id:
        item['label_id'] = '21'
    if 'orchard_vegetable' in image_id:
        item['label_id'] = '22'
    if 'pasture' in image_id:
        item['label_id'] = '23'
    if 'countryside' in image_id:
        item['label_id'] = '24'
    if 'greenhouse' in image_id:
        item['label_id'] = '25'
    if 'television_studio' in image_id:
        item['label_id'] = '26'
    if 'temple_east_asia' in image_id:
        item['label_id'] = '27'
    if 'pavilion' in image_id:
        item['label_id'] = '28'
    if 'tower' in image_id:
        item['label_id'] = '29'
    if 'palace' in image_id:
        item['label_id'] = '30'
    if 'church' in image_id:
        item['label_id'] = '31'
    if 'street' in image_id:
        item['label_id'] = '32'
    if 'dining_room' in image_id:
        item['label_id'] = '33'
    if 'coffee_shop' in image_id:
        item['label_id'] = '34'
    if 'kitchen' in image_id:
        item['label_id'] = '35'
    if 'plaza' in image_id:
        item['label_id'] = '36'
    if 'laboratory' in image_id:
        item['label_id'] = '37'
    if 'bar' in image_id:
        item['label_id'] = '38'
    if 'conference_room' in image_id:
        item['label_id'] = '39'
    if 'office' in image_id:
        item['label_id'] = '40'
    if 'hospital' in image_id:
        item['label_id'] = '41'
    if 'ticket_booth' in image_id:
        item['label_id'] = '42'
    if 'campsite' in image_id:
        item['label_id'] = '43'
    if 'music_studio' in image_id:
        item['label_id'] = '44'
    if 'elevator_staircase' in image_id:
        item['label_id'] = '45'
    if 'garden' in image_id:
        item['label_id'] = '46'
    if 'construction_site' in image_id:
        item['label_id'] = '47'
    if 'general_store' in image_id:
        item['label_id'] = '48'
    if 'clothing_store' in image_id:
        item['label_id'] = '49'
    if 'bazaar' in image_id:
        item['label_id'] = '50'
    if 'library_bookstore' in image_id:
        item['label_id'] = '51'
    if 'classroom' in image_id:
        item['label_id'] = '52'
    if 'ocean_beach' in image_id:
        item['label_id'] = '53'
    if 'firefighting' in image_id:
        item['label_id'] = '54'
    if 'gas_station' in image_id:
        item['label_id'] = '55'
    if 'landfill' in image_id:
        item['label_id'] = '56'
    if 'balcony' in image_id:
        item['label_id'] = '57'
    if 'recreation_room' in image_id:
        item['label_id'] = '58'
    if 'discotheque' in image_id:
        item['label_id'] = '59'
    if 'museum' in image_id:
        item['label_id'] = '60'
    if 'desert_sand' in image_id:
        item['label_id'] = '61'
    if 'raft' in image_id:
        item['label_id'] = '62'
    if 'forest' in image_id:
        item['label_id'] = '63'
    if 'bridge' in image_id:
        item['label_id'] = '64'
    if 'residential_neighborhood' in image_id:
        item['label_id'] = '65'
    if 'auto_showroom' in image_id:
        item['label_id'] = '66'
    if 'lake_river' in image_id:
        item['label_id'] = '67'
    if 'aquarium' in image_id:
        item['label_id'] = '68'
    if 'aqueduct' in image_id:
        item['label_id'] = '69'
    if 'banquet_hall' in image_id:
        item['label_id'] = '70'
    if 'bedchamber' in image_id:
        item['label_id'] = '71'
    if 'mountain' in image_id:
        item['label_id'] = '72'
    if 'station_platform' in image_id:
        item['label_id'] = '73'
    if 'lawn' in image_id:
        item['label_id'] = '74'
    if 'nursery' in image_id:
        item['label_id'] = '75'
    if 'beauty_salon' in image_id:
        item['label_id'] = '76'
    if 'repair_shop' in image_id:
        item['label_id'] = '77'
    if 'rodeo' in image_id:
        item['label_id'] = '78'
    if 'igloo_ice_engraving' in image_id:
        item['label_id'] = '79'
    result.append(item)

result = random.sample(result, len(result))

json.dump(result, open('scene_train_annotations_{0}.json'.format(name), 'w'))
print_list_info(result)
