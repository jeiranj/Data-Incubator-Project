from bs4 import BeautifulSoup
import urllib2
import mgrs
import csv
import re
import time
#sys.modules[__name__].__dict__.clear()
RATE_LIMIT_PER_SECONDS = 1.2
desired_keys = ['Title','Date','Lat','Lon','MGRS','Type','Category','Tracking number',
                'Region','Reporting unit','Unit name','Type of unit','Attack on',
                'Originator group', 'Updated by group','CCIR','Sigact','Affiliation',
                'Dcolor','Classification','Enemy detained',
                'Total casualties','Friendly wounded','Friendly killed',
                'Enemy wounded','Enemy killed','Civilian wounded','Civilian killed',
                'Host nation wounded','Host nation killed','Log']
        
class UrlRequester:
    def __init__(self):
        self.last_query_seconds = -1

        
    def rate_limit(self):
        time_since_last_query_seconds = time.time() - self.last_query_seconds
        if time_since_last_query_seconds < RATE_LIMIT_PER_SECONDS:
            #print 'last query '+ str(self.last_query_seconds)
            time.sleep(RATE_LIMIT_PER_SECONDS - time_since_last_query_seconds)
            #print 'waited '+ str(RATE_LIMIT_PER_SECONDS - time_since_last_query_seconds)
        self.last_query_seconds = time.time()


    def perform_url_request(self,url):
        # Try the URL request for 10 times, if successful quit. If unsuccessful return a bad result
        for i in range(0,30):
            try:
                # Rate limit only if doing a query
                self.rate_limit()
                response = urllib2.urlopen(url)
                results = response.read()
                return results
            except urllib2.HTTPError:
                print "Openning URL: " + url + " had an http error. Retrying...\n"
                time.sleep(20)
                continue
       
                
    def get_ids(self,page):    
        url = "http://warlogs.wikileaks.org/search/?release=Iraq&sort=date&p="+str(page)
        raw_page = self.perform_url_request(url)
        soup = BeautifulSoup(raw_page)
        page_results = soup.findAll('a',attrs={'class':'searchresult-title'})
        page_ids = [str(x.get('href')) for x in page_results]
        return page_ids
   

    def get_log(self,log_id):
        url="http://warlogs.wikileaks.org"+log_id
        raw_page = self.perform_url_request(url)
        soup = BeautifulSoup(raw_page)
        #print soup.prettify()
    
<<<<<<< HEAD
=======
>>>>>>> 20d5a91ce3f63a1a12da763eaa65a0d9bfbc702c
        parent_table = soup.find('table',attrs={'class':'metadata'})
        title = str(soup.find('h1',attrs={'class':'entry-title'}).text.encode('utf8', 'replace'))
        date = str(soup.find('div',attrs={'style':'clear: both;'}).text)
        date_pattern = re.compile(r'[\n\s](.+)[\n(\s+)]')
        date = date_pattern.findall(str(date))[0].strip()
        log = soup.find('noscript').text
        log = str(log.encode('utf8', 'replace'))
        log = re.sub("\nJavascript required for full view\nLimited script-free view:\n(\s+)",'', log)
    
        #Table keys:
<<<<<<< HEAD
        outcome = ['Enemy detained','Total casualties','Friendly wounded','Friendly killed',
                   'Enemy wounded','Enemy killed','Civilian wounded','Civilian killed',
                   'Host nation wounded','Host nation killed']
        table_th = parent_table.find_all('th')
        keys = [str(tmp.text) for tmp in table_th]
        keys.extend(outcome)
        
>>>>>>> 20d5a91ce3f63a1a12da763eaa65a0d9bfbc702c
        #Table values:
        table_td = parent_table.find_all('td')
        values = [str(tmp.text) for tmp in table_td]
        del tmp
        find = lambda values, elem: [[i for i, x in enumerate(values) if x == e] for e in elem]
        index_tmp = find(values, outcome)

        values_tmp = ['0'] * len(outcome)
<<<<<<< HEAD
        index_keys = []
        for i in range(len(index_tmp)):
            if index_tmp[i]:
                index_keys.extend(index_tmp[i])
                values_tmp[i] = values[index_tmp[i][0]+1]         
        if not index_keys == False:
            index_keys.extend([x+1 for x in index_keys])
            values = [item for index, item in enumerate(values) if index not in index_keys]
        values = values + values_tmp 
        table =  dict(zip(keys, values))
        
        #Get latitude and longitude from MGRS coordinates:
        try:
            lat, lon = self.mgrs_to_latLon(table['MGRS'])
        except mgrs.core.RTreeError:
            lat = ''
            lon = ''
            print 'MGRS problem at log'+log_id
        except ValueError:
            lat = ''
            lon = ''
            print 'Missing MGRS at log'+log_id
        
        #Add the other keys and their values
        table['Title'] = title
        table['Log'] = log
        table['Date'] = date
        table['Lat'] = str(lat)
        table['Lon'] = str(lon)
        values = list()
        for desired_key in desired_keys:
            my_value = ''
            for key, value in table.iteritems():
                if key == desired_key:
                    my_value = value
            values.append(my_value)
        del values_tmp, index_tmp, index_keys, table_th, table_td, key
        return values
        
        


>>>>>>> 20d5a91ce3f63a1a12da763eaa65a0d9bfbc702c
    def mgrs_to_latLon(self,mgrs_data):   
        m = mgrs.MGRS()
        mgrs_data = mgrs_data.replace(" ", "")
        lat, lon = m.toLatLon(mgrs_data)
        del m, mgrs_data
        return lat, lon


def main():
<<<<<<< HEAD
    last_page=7837#39184
    url_requester = UrlRequester()    
    page1 = 1
    with open('irq1.csv', 'w') as f_csv:
        writer = csv.writer(f_csv, delimiter=',')    
        writer.writerow(desired_keys)
        for page in range(page1,last_page+1):
            print '============Parsing page '+str(page)
            log_ids = url_requester.get_ids(page)
            f_csv.flush()
            for id in log_ids:
                values = url_requester.get_log(id)
                writer.writerow(values)
            
 
if __name__ == '__main__':
    main()
                   
