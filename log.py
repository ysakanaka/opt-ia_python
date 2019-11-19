import logging
import csv
import io
#import optIA


class CsvFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output)

    def format(self, record):
        self.writer.writerow([record.__dict__.get(
            'generation'), record.__dict__.get(
            'surplus_at_select')])
        #self.writer.writerow([record.levelname, record.msg,
        #                     record.__dict__.get('event', {})])
        #self.writer.writerow([record.msg])
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()


#data = {"event": "This is event data", "argtest": 2}

#logging.basicConfig(filename="logtest.csv", filemode='w', level=logging.INFO)

#logger = logging.getLogger(__name__)
#logging.root.handlers[0].setFormatter(CsvFormatter())

#logger.debug('This message should appear on the console')
#logger.info('This is test message the argument is %s', 2)
#logger.warning('And this, too')

#logger.info("dict info test", extra=dict(data))
