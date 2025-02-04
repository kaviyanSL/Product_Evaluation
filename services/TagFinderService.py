class TagFinderService ():
    def __init__(self,URL):
        self.URL = URL

    def readig_html_component(self):
        print("file readed")
        return (f"the URL in service is:{self.URL}")