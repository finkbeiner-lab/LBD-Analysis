import qupath.lib.objects.PathObjects
import qupath.lib.io.GsonTools
import groovy.io.FileType


def project = getProject()

for (entry in project.getImageList()){

    def name = entry.getImageName()
    print name
    def fileName = name + ".json"
    def pathInput = buildFilePath(PathPrefs.userPathProperty().get(), "regions", fileName)
    def imageData = entry.readImageData()
    
    def hierarchy = imageData.getHierarchy()
    def annotations = hierarchy.getAnnotationObjects()
    print annotations
    boolean prettyPrint = true
    def gson = GsonTools.getInstance(prettyPrint)
   
   def file = new File(pathInput)
   file.write(gson.toJson(annotations))
   
   }