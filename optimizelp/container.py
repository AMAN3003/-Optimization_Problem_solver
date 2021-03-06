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

    def __len__(self):
        return len(self.List_Objects)

    def __contains__(self, key):
        if key in self.Dict_Val:
            return True
        elif key in self.List_Objects:
            return True
        else:
            self.Re_Indexor()
            return key in self.Dict_Val

    def __iter__(self):
        original_length = len(self.List_Objects)
        for item in self.List_Objects.__iter__():
            if original_length != len(self.List_Objects):
                raise RuntimeError("container changed size during iteration")
            yield item

    def __getitem__(self, key):
        try:
            return self.List_Objects.__getitem__(key)
        except TypeError:
            try:
                return self.Dict_Val[key]
            except KeyError:
                self.Re_Indexor()
                try:
                    return self.Dict_Val[key]
                except KeyError:
                    raise KeyError("%s does not contain an object with name %s" % (self, key))

    def __setitem__(self, key, Var_Value):
        try:
            self.has_Name_Attribute(Var_Value)
            self.List_Objects.__setitem__(key, Var_Value)
            self.Dict_Val[Var_Value.name] = Var_Value
        except TypeError:
            try:
                item = self.Dict_Val.__getitem__(key)
                index = self.List_Names.index(item.name)
                self.Dict_Val[key] = Var_Value
            except KeyError:
                self.Re_Indexor()
                try:
                    self.Dict_Val[key] = Var_Value
                except KeyError:
                    raise KeyError("%s does not contain an object with name %s" % (self, key))
            self.List_Objects.__setitem__(index, Var_Value)

    def __delitem__(self, key):
        try:
            item = self.List_Objects.__getitem__(key)
            self.List_Objects.__delitem__(key)
            self.Dict_Val.__delitem__(item.name)
        except TypeError:
            try:
                item = self.Dict_Val.__getitem__(key)
                index = self.List_Names.index(item.name)
                self.Dict_Val.__delitem__(key)
            except KeyError:
                self.Re_Indexor()
                try:
                    self.Dict_Val.__delitem__(key)
                except KeyError:
                    raise KeyError("%s does not contain an object with name %s" % (self, key))
            self.List_Objects.__delitem__(index)

    def Iter_Keys(self):
        return self.List_Names.__iter__()

    def keys(self):
        return list(self.Iter_Keys())

    def Iter_Values(self):
        return self.List_Objects.__iter__()

    def values(self):
        return self.List_Objects

    def iteritems(self):
        for Element in self.List_Objects:
            yield Element.name, Element

    def From_Keys(self, keys):
        return self.__class__([self.__getitem__(key) for key in keys])

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except (KeyError, IndexError) as e:
            self.Re_Indexor()
            try:
                return self.__getitem__(key)
            except (KeyError, IndexError) as e:
                return default

    def clear(self):
        self.List_Objects = list()
        self.Dict_Val = dict()

    def has_key(self, key):
        if key in self.Dict_Val:
            return True

    def append(self, Var_Value):
        self.has_Name_Attribute(Var_Value)
        name = Var_Value.name
        if name in self.Dict_Val:
            raise Exception("Container '%s' already contains an object with name '%s'." % (self, Var_Value.name))
        self.List_Objects.append(Var_Value)
        self.Dict_Val[Var_Value.name] = Var_Value

    def extend(self, values):
        for Var_Value in values:
            self.has_Name_Attribute(Var_Value)
            if Var_Value.name in self.Dict_Val:
                raise Exception("Container '%s' already contains an object with name '%s'." % (self, Var_Value.name))
        self.List_Objects.extend(values)
        self.Dict_Val.update(dict([(Var_Value.name, Var_Value) for Var_Value in values]))

    def __getattr__(self, name):
        try:
            return self.__getitem__(name)
        except KeyError:
            raise AttributeError("'%s' object has no attribute %s" % (self, name))

    def __getstate__(self):
        return self.List_Objects

    def __setstate__(self, obj_list):
        self.__init__(obj_list)

    def __dir__(self):
        attributes = list(self.__class__._Dict_Val__.keys())
        attributes.extend(self.List_Names)
        return attributes