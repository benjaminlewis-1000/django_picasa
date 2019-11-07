
# import re
# from datetime import datetime
# from django.utils import timezone

# from io import BytesIO
# from django.core.files.base import ContentFile
# import os
# import time
# from django.core.files.uploadedfile import SimpleUploadedFile
# import hashlib
# import PIL
# from PIL.ExifTags import TAGS, GPSTAGS
# from PIL import Image
# from datetime import datetime
# from django.core.files import File
# import pytz
# import logging
# import cv2
# import numpy as np

# # Image thumbnail processing
# from stdimage import StdImageField, JPEGField
# from stdimage.utils import render_variations

# datetime.utcnow().replace(tzinfo=pytz.utc)

# image = PIL.Image.open('/home/benjamin/git_repos/picasa_files/test_imgs/2019-11-03 16.44.50.jpg')
# exifDict = {}
# # print(filename)
# try:
#     info = image._getexif()
# except AttributeError as ae:
#     info = None
# if info is not None:
#     for tag, value in info.items():
#         decoded = TAGS.get(tag, tag)
#         exifDict[decoded] = value
# else:
#     orientation = 1
#     exifDict = None


#     # Default comparison date - we want the earliest date.
# dateTaken = datetime.now()

# # unaware = datetime.datetime(2011, 8, 15, 8, 15, 12, 0)
# # aware = datetime.datetime(2011, 8, 15, 8, 15, 12, 0, pytz.UTC)

# dateTaken = pytz.utc.localize(dateTaken)
# print(dateTaken)
# # Flag for if we found a date taken in the EXIF data. 
# dateTakenValid = False

# if exifDict is not None:
#     dateTakenKeys = ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']
#     for exifKey in dateTakenKeys:
#         datetaken_tmp = exifDict[exifKey] if exifKey in exifDict.keys() else None
#         if datetaken_tmp is None:
#             continue  # No value at this EXIF key
#         else:
#             date = datetime.strptime(datetaken_tmp, '%Y:%m:%d %H:%M:%S')
#             date = pytz.utc.localize(date)
#             print(exifKey)
#             print(date)
#             if date < dateTaken: 
#                 dateTaken = date
#                 dateTakenValid = True
# else:
#     dateTaken = datetime.now()
#     dateTakenValid = False


    #! /usr/bin/python

import pyexiv2
import os
import re
import sys
from datetime import datetime
from dateutil import parser
import json
import xmltodict
import xmlrpc
from time import sleep

import socket
socket.setdefaulttimeout(4)        #set the timeout to 10 second

project_path = os.path.abspath(os.path.join(__file__,"../.."))
script_path  = os.path.abspath(os.path.join(__file__,".."))

# with open(project_path + '/config/params.xml') as stream:
#     try:
#         params = xmltodict.parse(stream.read())
#     except Exception as exc:
#         print(exc)
#         exit(1)

# desiredField = "Exif.Photo.DateTimeOriginal"

'''
Fields to get:
= Names (and areas?)
= File creation date
na Modification date
na FileSize
na Image Width
na Image Height
= GPS Lat
= GPS Long
- Keywords (See classImage)
= Time zone (where applicable)
'''

''' 
getExif is a function that retrieves, where avaliable, the following data from a JPEG image: 

'ISO8601' - ISO format date/time of the photo
'year' - the respective value from the ISO date
'month' - the respective value from the ISO date
'day' - the respective value from the ISO date
'hour' - the respective value from the ISO date
'min' - the respective value from the ISO date
'sec' - the respective value from the ISO date
'TimeZone' - the respective value from the ISO date

'latitude' - lat of the photo, if set
'longitude' - lon of the photo, if set

'house_number' - decoded from lat/lon if doGeocode is set to "True"
'road' - decoded from lat/lon if doGeocode is set to "True"
'city' - decoded from lat/lon if doGeocode is set to "True"
'state' - decoded from lat/lon if doGeocode is set to "True"
'postcode' - decoded from lat/lon if doGeocode is set to "True"
'country' - decoded from lat/lon if doGeocode is set to "True"

'names' - a list of all names of people in the photo, de-duplicated
'picasaTags' - tags that are user-set, from Picasa
'autoTagsClarifai' - tags that were automatically assigned from Clarifai
'autoTagsGoogle' - tags that were automatically assigned from Google services

'''

def getExifData(filename, doGeocode):

    if filename.endswith(tuple([".JPG", ".jpg", ".jpeg", ".JPEG"])):
        # print filename
       
        # Read the exiv2 metadata from the image
        metadata = pyexiv2.ImageMetadata(filename)
        try:
            metadata.read()
        except IOError as ioe:
            print ("IO Error in getExif.py: " +  str(ioe))
            logfile = open(script_path + '/logBadMetadata.out', 'a')
            # print >>logfile, str(ioe) + ": metadata error in " + filename 
            logfile.close()
            # Return null values
            return "", ""
        dateOK = True
        jData = {}

        # Preferred fields: 'Exif.Photo.DateTimeOriginal' or 'Exif.Photo.DateTimeDigitized'
        # Retrieve the time from the EXIV

        dateIso = 'n/a'
        dateOK = False
        def extract_date(date_string):

            print( date_string)
            if " " in date_string:
                date, time = date_string.split(" ")
            if len(date_string) > 9 and "T" in date_string[10] and '-' in date_string:
                date, time = date_string.split("T")
            date = date.replace(':', '-')
            date_string = date + " " + time
            #print "Date string: " +  date_string
            try:
                dateTime = parser.parse(date_string)
                dateTime = dateTime.strftime("%Y-%m-%d %H:%M:%S")
                dateOK = True
            except ValueError:
                dateTime = 'n/a'
                dateOK = False

            #print dateTime
            return dateTime, dateOK

        # Area 1: 'Exif.Photo.DateTimeOriginal'
        if 'Exif.Photo.DateTimeOriginal' in metadata:
            metadata_date = metadata["Exif.Photo.DateTimeOriginal"].raw_value
            dateIso, dateOK = extract_date(metadata_date)

        if not dateOK and 'Exif.Photo.DateTimeDigitized' in metadata:
            metadata_date = metadata["Exif.Photo.DateTimeDigitized"].raw_value
            dateIso, dateOK = extract_date(metadata_date)

        if not dateOK and 'Iptc.Application2.DateCreated' in metadata and 'Iptc.Application2.TimeCreated' in metadata:
            date = metadata['Iptc.Application2.DateCreated'].raw_value[0]
            time = metadata['Iptc.Application2.TimeCreated'].raw_value[0]
            metadata_date = str(date) + " " + str(time)
            dateIso, dateOK = extract_date(metadata_date)

        if not dateOK:
            dateIso = 'a'
            logfile = open(script_path + '/logNoDates.out', 'a')
            # print >>logfile, "Question the time for: " + filename 
            logfile.close()

#        print dateIso
        ## See if the file name is any good:
        file_name_string = filename.split('/')[-1].split('\\')[-1]

        for ending in tuple([".JPG", ".jpg", ".jpeg", ".JPEG"]):
            file_name_string = file_name_string.replace(str(ending), "")

        # Common format seems to be yyyy-mm-dd hh.mm.ss
        if re.search('\d{4}.\d{2}.\d{2}.\d{2}.\d{2}.\d{2}', file_name_string):
            file_name_string = file_name_string.replace('.', ':')
            file_name_string = file_name_string.replace('-', ':')
            file_name_string = file_name_string[:19]
            try:
                fileDateIso, fileDateOK = extract_date(file_name_string)
            except UnboundLocalError as ule:
                fileDateOK = False
            if fileDateOK and fileDateIso < dateIso:
                dateIso = fileDateIso
                dateOK = fileDateOK


        if not dateOK:
            print ("Entering Date Taken Remediation. ")
            # print ""
            # print ""
            ## Try to find an acceptable date. Start with a non-date value that all numbers
            ## will be less than so that we get the earliest date. 
            dateIso = "a"
            for key in metadata.keys():
                if re.search('date', key, re.IGNORECASE):
                    rawTime = metadata[key].raw_value
                    # print "raw time: " + str(rawTime)

                    possVal, isOK = extract_date(rawTime)
                    if isOK and possVal < dateIso:
                        dateIso = possVal

            if dateIso == 'a':
                # Last resort failsafe -- none of the above worked. 
                dateIso = '1900:01:01 00:00:01'



            # print "DateISO is " + dateIs
        print ("Date iso is : " + str(dateIso) + " on file: " + filename)

        dt = parser.parse(dateIso)

        jData['ISO8601'] = dateIso
        jData['year'] = dt.year
        jData['month'] = dt.month
        jData['day'] = dt.day
        jData['hour'] = dt.hour
        jData['min'] = dt.minute
        jData['sec'] = dt.second

        if 'Exif.NikonWt.Timezone' in metadata:
            jData['TimeZone'] = metadata['Exif.NikonWt.Timezone'].raw_value
        else:
            jData['TimeZone'] = '0'

        # Get the names of tagged faces
        etadataFields = metadata.xmp_keys
        # nameList = []
        # print metadataFields

        # There are two different ways of storing names in XMP data, so I do 
        # both of them and join them in a set list.
        regionlist_name_regex = re.compile('.*RegionList.*Name.*')
        nameKeys = filter(regionlist_name_regex.match, metadata.keys())
        nameList = [metadata[nk].value for nk in nameKeys]
        # Take out the JPEG filenames in the 'name' list - happens sometimes apparently.
        jpeg_regex = re.compile('.*JPE?G$', re.IGNORECASE)
        jpeg_list = filter(jpeg_regex.search, nameList)
        # Also takes out some bogus names...
        nameList = set(nameList) - set(jpeg_list) - set(["Custom", "Medium Contrast"])
        # print nameList

        # for i in range(len(metadataFields)):
        #     # Find all fields that have the word 'name' in them, regardless of case. 
        #     if re.search(r'Name', metadataFields[i], re.IGNORECASE):
        #         nameSplit = metadata[metadataFields[i]].raw_value
        #         nameSplit = nameSplit.split(": ")
        #         name = nameSplit[0]
        #         bogusNames = ["Custom", "Medium Contrast"]
        #         if not re.search(r'^\.', name) and not name.endswith(tuple([".JPG", ".jpg", ".jpeg", ".JPEG"])) and not name in bogusNames and not re.search(r'NIKKOR', name) and not re.search(r'NIKON', name):
        #             # Not ending in JPEG because we get the filename from this list in the pitc values sometimes. 
        #             name = name.rstrip()
        #             name = ' '.join([s[0].upper() + s[1:] for s in name.split(' ')])
        #             nameList.append(name)
        # print nameList
        jData['names'] = list(nameList) # .union(set(nameList)))  ## De-duplicate names


        # if 'Exif.Photo.UserComment' in metadata:
        #     autoTags = metadata['Exif.Photo.UserComment'].raw_value
        #     # autoTags = re.sub(r'charset=\"Ascii\"\s+', '', autoTags)
        #     if re.search(r'UUUUUUUU', autoTags):
        #         # Reject a tag that has been shown to have a erroneous value in these fields.
        #         jData['autoTagsGoogle'] = []
        #         jData['autoTagsClarifai'] = []
        #     else:
        #         tagSplit = autoTags.split(', ')
        #         # print tagSplit
        #         autoTagsGoogList = []
        #         autoTagsClarList = []
        #         for tagOrig in tagSplit:
        #             googVisionPrefix = params['params']['visionTaggingParams']['googleTagging']['googVisionLabelPrefix']
        #             clarifaiPrefix = params['params']['visionTaggingParams']['clarifaiTagging']['clarifaiVisionLabelPrefix']
        #             tag = re.sub(r'' + googVisionPrefix, '', tagOrig)
        #             tag = re.sub(r'' + clarifaiPrefix, '', tag)
        #             if re.search(r'_', tag):
        #                 try:
        #                     (comment, val) = tag.split('_')
        #                 except ValueError as ve:
        #                     logfile = open(os.path.join(script_path, 'tagSplitErrors.out'), 'a')
        #                     print >>logfile, 'Tag was ill-formed in file ' + str(filename) + ' ' + tag
        #                     continue
        #                 try:
        #                     comment = comment.decode('utf-8')
        #                 except UnicodeDecodeError as ude:
        #                     print comment
        #                     print "Unicode Error in this file: " + str(filename)

        #                     logfile = open(script_path + '/unicodeTagErrors.out', 'a')
        #                     print >>logfile, "Unicode Tag Error in this file: " + str(filename)
        #                     # exit(1)
        #                 tagPair = (comment, val)
        #                 if not (re.search('METADATA-START', tagOrig) or (tagOrig in nameList)):
        #                     # print tagPair[0]
        #                     if re.search(googVisionPrefix, tagOrig):
        #                         autoTagsGoogList.append(tagPair)
        #                     if re.search(clarifaiPrefix, tagOrig):
        #                         autoTagsClarList.append(tagPair)
        #         jData['autoTagsGoogle'] = autoTagsGoogList
        #         jData['autoTagsClarifai'] = autoTagsClarList
        #         # assert len(jData['autoTags']) > 0
        #         # assert not re.search(r'\\x[0-9a-f]{2}', str(autoTagsGoogList))
        #         # assert not re.search(r'\\x[0-9a-f]{2}', str(autoTagsClarifai))
        #         assert not re.search(r'googVision_', str(autoTagsGoogList))
        #         assert not re.search(r'clarifaiVision_', str(autoTagsClarList))
        # else:
        #     jData['autoTagsGoogle'] = []
        #     jData['autoTagsClarifai'] = []

        # if 'Iptc.Application2.Keywords' in metadata:
        #     picasaTags = metadata['Iptc.Application2.Keywords'].raw_value
        #     # tagSplit = picasaTags.split(',')
        #     pTags = []
        #     for tagOrig in picasaTags:
        #         if not (re.search('METADATA-START', tagOrig) or (tagOrig in nameList)):
        #             pTags.append( (tagOrig, 1) )
        #     jData['picasaTags'] = pTags
        #     # assert len(jData['picasaTags']) > 0
        # else:
        #     jData['picasaTags'] = []

        # # Don't process pictures that are tagged as 'exclude'
        # picTagStrings = [val[0].lower() for val in jData['picasaTags']]
        # if set(['exclude', 'ignore', 'remove']).intersection(set(picTagStrings)) != set([]):
        #     if __name__ == "__main__":
        #         return -1, -1
        #     else:
        #         return -1

        jData['latitude'] = "-"
        jData['longitude'] = "-"
        jData['house_number'] = "-"
        jData['road'] = "-"
        jData['city'] = "-"
        jData['state'] = "-"
        jData['postcode'] = "-"
        jData['country'] = "-"

        if ('Exif.GPSInfo.GPSLatitudeRef' in metadata):
            # Get the latitude/longitude and N/S and E/W
            northSouth = metadata['Exif.GPSInfo.GPSLatitudeRef'].raw_value
            eastWest = metadata['Exif.GPSInfo.GPSLongitudeRef'].raw_value
            lat = str(metadata['Exif.GPSInfo.GPSLatitude'].raw_value)
            lon = str(metadata['Exif.GPSInfo.GPSLongitude'].raw_value)
            latSplit = lat.split(" ")
            lonSplit = lon.split(" ")
            assert len(latSplit) == 3
            assert len(lonSplit) == 3
            # Compute the latitude and longitude in decimal form
            latDec = 1.0 * eval(latSplit[0] + '.0') + 1.0/60.0 * eval(latSplit[1] + '.0') + 1.0/3600.0 * eval(latSplit[2] + '.0')
            lonDec = 1.0 * eval(lonSplit[0] + '.0') + 1.0/60.0 * eval(lonSplit[1] + '.0') + 1.0/3600.0 * eval(lonSplit[2] + '.0')

            if (northSouth != "N"):
                latDec *= -1.0
            if (eastWest != "E"):
                lonDec *= -1.0

            jData['latitude'] = latDec
            jData['longitude'] = lonDec

            # if doGeocode:
            #     # Do a reverse geocode lookup using the geoServer.py server, which is started by addPics.py
            #     passed = False
            #     proxy = xmlrpclib.ServerProxy("http://127.0.0.1:" + params['params']['serverParams']['geoServerPort'] + "/RPC2")
            #     try:
            #         val = proxy.geoLookup(latDec, lonDec)
            #         if val != -1:
            #         #    if __name__ == "__main__":
            #         #        return -1, -1
            #         #    else:
            #         #        return -1
            #         #else:
            #             try:
            #                 val = json.loads(val.encode('utf-8'))
            #             except Exception as e :
            #                 print("No JSON Object...")
            #             passed = True
            #     except Exception as e :
            #         return str(e)

            #     if passed:

            #         jData['house_number'] = val['house_number']
            #         jData['road'] = val['road']
            #         jData['city'] = val['city']
            #         jData['state'] = val['state']
            #         jData['postcode'] = val['postcode']
            #         jData['country'] = val['country']
            #         # $jsonData->{'postcode'} =~ m/(\d+)/g;
            #         # my $postcode = $1;


        jsonObj = json.dumps(jData)
        assert re.search(r'...............', jsonObj)
        # print __name__
        if __name__ == "__main__":
            return jsonObj, metadata
        else:
            return jsonObj


if __name__ == '__main__':


    fullPath = "D:\\Pictures\\Emily's Pictures\\3-30-2009\\DSCN0456.JPG"
    fullPath = '/home/benjamin/git_repos/picasa_files/test_imgs/2019-11-03 16.44.50.jpg'
    jsonObj, metadata = getExifData(fullPath, False)
    print(jsonObj)




#     dirFile = open(script_path + "/directories.txt", 'r')
#     directories = dirFile.read().splitlines()
#     for curDir in directories:
#         print curDir
#         for filenames in os.listdir(curDir):
#             fullPath = os.path.join(curDir, filenames)
#             print fullPath
#             print getExifData(fullPath, True)
#             for fname in filenames:
#                 if fname.endswith(tuple([".JPG", ".jpg", ".jpeg", ".JPEG"])):
#                     filename = os.path.join(dirpath, fname)
#                     # getExifData(filename, False)
#                     print filename