import logging
import csv
import io


class CsvFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output, quoting=csv.QUOTE_NONNUMERIC)

    def format(self, record):
        self.writer.writerow([record.msg])
        #self.writer.writerow([record.msg])
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()

#logging.basicConfig(filename="logtest.csv", level=logging.DEBUG)

#logger = logging.getLogger(__name__)
#logging.root.handlers[0].setFormatter(CsvFormatter())

#logger.debug('This message should appear on the console')
#logger.info('So should "this", and it\'s using quoting...')
#logger.warning('And this, too')