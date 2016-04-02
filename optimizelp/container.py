class Container(object):
    '''A container for objects that have a name attribute.'''

    def __init__(self, iterable=[]):
        try:
            self.Dict_Val = dict(((Element.name, Element) for Element in iterable))
        except AttributeError as e:
            print("Only objects with containing a 'name' attribute can be stored in a Container.")
            raise e
        self.List_Objects = list(iterable)

    @property
    def List_Names(self):
        return [Element.name for Element in self.List_Objects]

    @staticmethod
    def has_Name_Attribute(Var_Value):
        if not hasattr(Var_Value, 'name'):
            raise AttributeError('Object %s does not have a "name" attribute and cannot not be stored.' % Var_Value)

    def Re_Indexor(self):
        self.Dict_Val = dict(((Element.name, Element) for Element in self.List_Objects))
