import java.awt.image.BufferedImage
import javafx.beans.value.ChangeListener
import javafx.beans.value.ObservableValue
import qupath.lib.gui.scripting.DefaultScriptEditor
import qupath.lib.gui.QuPathGUI
import qupath.lib.projects.Project
import qupath.lib.projects.ResourceManager
import java.io.File

import qupath.lib.common.ColorTools
import qupath.lib.objects.classes.PathClassFactory
import qupath.lib.gui.scripting.QPEx
import qupath.lib.gui.QuPathGUI
import qupath.lib.gui.prefs.PathPrefs
import qupath.lib.gui.tools.GuiTools
import javafx.application.Platform
import qupath.lib.gui.dialogs.Dialogs


import groovy.io.FileType
import java.awt.image.BufferedImage
import qupath.lib.images.servers.ImageServerProvider
import qupath.lib.gui.commands.ProjectCommands
import qupath.lib.gui.prefs.PathPrefs
import groovy.io.FileType


def pathClassGroup = "LBD"
def pathClassMaps = [
    "Amyloid staining": [
        "Cored:CTRL+C": [0, 255, 255],
        "Diffuse:CTRL+F": [255, 153, 102],
        //"CAA: CTRL+A": [150, 79, 239],
        "Coarse-Grained:CTRL+R":[255, 0, 239],
        "Cotton-Wool:CTRL+W":[160,170,90],
        "Burned-Out:CTRL+B":[124,125,255],
        "DELETE-Annotation:DELETE":[0,0,0]
    ],
   "AT8": [
        "Pre:CTRL+P": [0, 255, 0],
        "Mature:CTRL+M": [0, 0, 255],
        "Ghost:CTRL+G": [255, 0, 0],
        "DELETE-Annotation:CTRL+D":[0,0,0]
    ],
    "Biel":[
         "Cored:CTRL+C": [0, 255, 255],
        "Diffuse:CTRL+F": [255, 153, 102],
        //"CAA: CTRL+A": [150, 79, 239],
        "Coarse-Grained:CTRL+R":[255, 0, 239],
        "Cotton-Wool:CTRL+W":[160,170,90],
        "Burned-Out:CTRL+B":[124,125,255],
        "Pre:CTRL+P": [0, 255, 0],
        "Mature:CTRL+M": [0, 0, 255],
        "Ghost:CTRL+G": [255, 0, 0],
        "DELETE-Annotation:CTRL+D":[0,0,0]
    ],
    "LBD":[
        //"Pre LB:CTRL+P": [0, 255, 0],
        //"Mature LB:CTRL+M": [0, 0, 255],
        "LB:CTRL+B": [0, 0, 255],
        "LN:CTRL+N": [0, 0, 255],
        "DELETE-Annotation:CTRL+D":[0,0,0]

    ]
    
   ]
   
def prefMap = [
    "gridScaleMicrons": (boolean) true,
    "gridSpacingX": (double) 1024,
    "gridSpacingY": (double) 1024,
    "gridStartX": (double) 0,
    "gridStartY": (double) 0,

    "multipointTool": (boolean) false,
]


class ScriptLoader {
    static File scriptsPath = new File(new File(PathPrefs.userPathProperty().get()), "scripts")

    def ScriptLoader() {
        assert this.scriptsPath.isDirectory()
    }

    def getScriptFile(String path) {
        return new File(this.scriptsPath, path)
    }

    def getScript(String path) {
        def scriptFile = this.getScriptFile(path)
        assert scriptFile.isFile()
        return Eval.me(scriptFile.getText())
    }

    def getScriptOptional(String path) {
        def scriptFile = this.getScriptFile(path)
        return scriptFile.isFile() ? Eval.me(scriptFile.getText()) : null
    }
}

def setPrefs(Map<String, Object> map) {
    return map
        .collectEntries({[it.key, PathPrefs.@"${it.key}"]})
        .findAll({it.value.get() != map.get(it.key)})
        .each({it.value.set(map.get(it.key))})
}


def setPathClasses(QuPathGUI gui, Map<String, List<Integer>> map) {
    Platform.runLater({
        gui.getAvailablePathClasses().setAll([
            PathClassFactory.getPathClassUnclassified(),
            *map.collect({
                assert it.value.size() == 3
                def pathClass = PathClassFactory.getPathClass(it.key)
                pathClass.setColor(ColorTools.packRGB(*it.value))
                return pathClass
            })
        ])
    })
}


def build_project() {
    File directory = new File(PathPrefs.userPathProperty().get() + "/project/")
    
    File projectFile = new File(PathPrefs.userPathProperty().get() + "/project/project.qpproj")
    print projectFile
    if (projectFile.exists()){
          print "Project Exists"
          return false
     }

    File selectedDir = new File(PathPrefs.userPathProperty().get()+"/data/")
    
    def project = Projects.createProject(directory, BufferedImage.class)
    


    def files = []
    selectedDir.eachFileRecurse (FileType.FILES) { file ->
        if (file.getName().toLowerCase().endsWith(".svs"))
        {
            files << file
            print(file.getCanonicalPath())      
        }
    }

    double newPixelWidth = 1
    double newPixelHeight = 1

    for (file in files) {
        def imagePath = file.getCanonicalPath()
        //def support = ImageServerProvider.getPreferredUriImageSupport(BufferedImage.class, file.toURI().toString())
        
        def support = ImageServerProvider.getPreferredUriImageSupport(BufferedImage.class, imagePath)
    
        print support
        def builder = support.builders.get(0)
        
            print "Adding: " + imagePath
        entry = project.addImage(builder)
        
        // Set a particular image type
        def imageData = entry.readImageData()
        imageData.setImageType(ImageData.ImageType.BRIGHTFIELD_H_DAB)
        //imageData.setPixelCalibration(newPixelWidth, newPixelHeight)
        entry.setImageName(file.getName())
        def name = entry.getImageName()
        def fileName = name + ".json"
        
        entry.saveImageData(imageData)
        
        def pathInput = buildFilePath(PathPrefs.userPathProperty().get(), "regions", fileName)
        def gson = GsonTools.getInstance(true)
        def json = new File(pathInput).text
        def type = new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>() {}.getType();
    
        //print type
        def deserializedAnnotations = gson.fromJson(json, type);
        
        //def imageData = entry.readImageData()
        
        def hierarchy = imageData.getHierarchy()
        
        print deserializedAnnotations
        hierarchy.addObjects(deserializedAnnotations)  
        entry.saveImageData(imageData)
        
    }
    
    project.syncChanges()

    return true
    }





gui = QuPathGUI.getInstance()
setPrefs(prefMap)
setPathClasses(gui, pathClassMaps.get(pathClassGroup))



def build_project = build_project()





gui.imageDataProperty().addListener(new ChangeListener<ImageData<BufferedImage>>() {
    @Override
    void changed(ObservableValue<? extends ImageData<BufferedImage>> observable, ImageData<BufferedImage> oldProject, ImageData<BufferedImage> newProject) {
        if (newProject == null) {
            return
         }
        //def manager = newProject.getScripts()
        //print manager
        //def script
        //print newProject.currentLanguageProperty()
        try {
            def contextMenu = (new ScriptLoader()).getScript("main.groovy")
            //def importRegions = (new ScriptLoader()).getScript("import_regions.groovy")
             
        } catch (IOException ignored) {
            return
        }
        //DefaultScriptEditor.executeScript(DefaultScriptEditor.Language.GROOVY, contextMenu, newProject, null, true, null)
    //}
    }
})



println("Registered per project 'startup.groovy' handler")












