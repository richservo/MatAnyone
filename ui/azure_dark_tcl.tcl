# azure-dark.tcl - A dark theme for ttk based on Azure
# A dark modern theme inspired by Microsoft's Azure style
# Created by Claude for MatAnyone GUI

package require Tk 8.6

namespace eval ttk::theme::azure-dark {

    variable colors
    array set colors {
        -fg             "#ffffff"
        -bg             "#2d2d30"
        -disabledfg     "#7a7a7a"
        -disabledbg     "#2d2d30"
        -selectfg       "#ffffff"
        -selectbg       "#007acc"
        -window         "#252526"
        -focuscolor     "#1e88e5"
        -checklight     "#333333"
        -accent         "#007acc"
        -accent2        "#0063b1"
        -border         "#3f3f46"
        -darkbg         "#252526"
        -framebg        "#333337"
        -hovercolor     "#3e3e42"
    }

    proc LoadImages {imgdir} {
        variable I
        foreach file [glob -directory $imgdir *.png] {
            set img [file tail [file rootname $file]]
            set I($img) [image create photo -file $file -format png]
        }
    }

    # Create theme
    ttk::style theme create azure-dark -parent default -settings {
        ttk::style configure . \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -troughcolor $colors(-bg) \
            -focuscolor $colors(-focuscolor) \
            -selectbackground $colors(-selectbg) \
            -selectforeground $colors(-selectfg) \
            -insertcolor $colors(-fg) \
            -insertwidth 1 \
            -fieldbackground $colors(-window) \
            -font "TkDefaultFont" \
            -borderwidth 1 \
            -relief flat

        ttk::style map . -foreground [list disabled $colors(-disabledfg)]
        ttk::style map . -background [list disabled $colors(-disabledbg)]
        
        # Button
        ttk::style configure TButton \
            -anchor center -width -10 -padding {5 2} \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -borderwidth 1 -relief flat
            
        ttk::style map TButton \
            -background [list active $colors(-hovercolor) pressed $colors(-accent2)] \
            -relief [list {pressed !disabled} sunken] \
            -bordercolor [list active $colors(-accent)] \
            -foreground [list active $colors(-fg) disabled $colors(-disabledfg)]
            
        # Accent button style
        ttk::style configure Process.TButton \
            -anchor center -width -15 -padding {8 4} \
            -background $colors(-accent) \
            -foreground white
            
        ttk::style map Process.TButton \
            -background [list active $colors(-accent2) pressed "#00559e"] \
            -foreground [list active white disabled $colors(-disabledfg)]

        # Toolbutton
        ttk::style configure Toolbutton \
            -anchor center \
            -padding 1 \
            -relief flat \
            -background $colors(-bg)
            
        ttk::style map Toolbutton \
            -background [list active $colors(-hovercolor) pressed $colors(-accent2)] \
            -foreground [list active $colors(-fg) disabled $colors(-disabledfg)]

        # Entry
        ttk::style configure TEntry \
            -foreground $colors(-fg) \
            -background $colors(-window) \
            -fieldbackground $colors(-window) \
            -selectbackground $colors(-selectbg) \
            -selectforeground $colors(-selectfg) \
            -borderwidth 1 \
            -relief solid
            
        ttk::style map TEntry \
            -fieldbackground [list readonly $colors(-bg) disabled $colors(-disabledbg)] \
            -bordercolor [list focus $colors(-accent)] \
            -lightcolor [list focus $colors(-accent)] \
            -selectbackground [list !focus "#454545"] \
            -selectforeground [list !focus $colors(-fg)]

        # Combobox
        ttk::style layout TCombobox {
            Combobox.field -sticky nswe -children {
                Combobox.downarrow -side right -sticky ns
                Combobox.padding -expand 1 -sticky nswe -children {
                    Combobox.textarea -sticky nswe
                }
            }
        }
        
        ttk::style configure TCombobox \
            -foreground $colors(-fg) \
            -background $colors(-window) \
            -fieldbackground $colors(-window) \
            -selectbackground $colors(-selectbg) \
            -selectforeground $colors(-selectfg) \
            -borderwidth 1 \
            -padding 1 \
            -relief solid
            
        ttk::style map TCombobox \
            -fieldbackground [list readonly $colors(-bg) disabled $colors(-disabledbg)] \
            -bordercolor [list focus $colors(-accent)] \
            -lightcolor [list focus $colors(-accent)] \
            -selectbackground [list !focus "#454545"] \
            -selectforeground [list !focus $colors(-fg)] \
            -foreground [list disabled $colors(-disabledfg)]

        # Spinbox
        ttk::style configure TSpinbox \
            -foreground $colors(-fg) \
            -background $colors(-window) \
            -fieldbackground $colors(-window) \
            -selectbackground $colors(-selectbg) \
            -selectforeground $colors(-selectfg) \
            -borderwidth 1 \
            -relief solid \
            -arrowsize 12
            
        ttk::style map TSpinbox \
            -fieldbackground [list readonly $colors(-bg) disabled $colors(-disabledbg)] \
            -selectbackground [list !focus "#454545"] \
            -selectforeground [list !focus $colors(-fg)] \
            -foreground [list disabled $colors(-disabledfg)]

        # Label frames
        ttk::style configure TLabelframe \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -borderwidth 1 \
            -relief groove
        
        ttk::style configure TLabelframe.Label \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -padding {5 2}

        # Notebook
        ttk::style configure TNotebook \
            -background $colors(-darkbg) \
            -borderwidth 0 \
            -tabmargins {2 2 2 0}
            
        ttk::style configure TNotebook.Tab \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -padding {6 2} \
            -borderwidth 1
            
        ttk::style map TNotebook.Tab \
            -background [list selected $colors(-accent) active $colors(-hovercolor)] \
            -foreground [list selected white] \
            -padding [list selected {6 2}]

        # Labelframe
        ttk::style configure TLabelframe \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -borderwidth 1 \
            -relief solid \
            -bordercolor $colors(-border)
            
        ttk::style configure TLabelframe.Label \
            -background $colors(-bg) \
            -foreground $colors(-fg)

        # Scrollbar
        ttk::style layout Vertical.TScrollbar {
            Vertical.Scrollbar.trough -sticky ns -children {
                Vertical.Scrollbar.thumb -expand true
            }
        }
        
        ttk::style layout Horizontal.TScrollbar {
            Horizontal.Scrollbar.trough -sticky we -children {
                Horizontal.Scrollbar.thumb -expand true
            }
        }
        
        ttk::style configure TScrollbar \
            -background $colors(-darkbg) \
            -borderwidth 0 \
            -troughcolor $colors(-darkbg) \
            -arrowcolor $colors(-fg)
            
        ttk::style map TScrollbar \
            -background [list hover $colors(-hovercolor) active $colors(-accent) disabled $colors(-bg)] \
            -troughcolor [list !disabled $colors(-darkbg)]

        # Scale
        ttk::style configure TScale \
            -background $colors(-bg) \
            -troughcolor $colors(-darkbg) \
            -borderwidth 0
            
        ttk::style map TScale \
            -background [list active $colors(-accent)]

        # Progressbar
        ttk::style configure TProgressbar \
            -troughcolor $colors(-darkbg) \
            -background $colors(-accent)

        # Checkbutton
        ttk::style configure TCheckbutton \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -indicatorbackground $colors(-window)
            
        ttk::style map TCheckbutton \
            -background [list active $colors(-hovercolor)] \
            -foreground [list disabled $colors(-disabledfg)] \
            -indicatorbackground [list pressed $colors(-accent2) {!disabled active} $colors(-hovercolor) disabled $colors(-disabledbg)] \
            -indicatorcolor [list selected $colors(-accent)]

        # Radiobutton
        ttk::style configure TRadiobutton \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -indicatorbackground $colors(-window)
            
        ttk::style map TRadiobutton \
            -background [list active $colors(-hovercolor)] \
            -foreground [list disabled $colors(-disabledfg)] \
            -indicatorbackground [list pressed $colors(-accent2) {!disabled active} $colors(-hovercolor) disabled $colors(-disabledbg)] \
            -indicatorcolor [list selected $colors(-accent)]

        # Treeview
        ttk::style configure Treeview \
            -background $colors(-window) \
            -foreground $colors(-fg) \
            -fieldbackground $colors(-window) \
            -borderwidth 0 \
            -relief flat
            
        ttk::style map Treeview \
            -background [list selected $colors(-selectbg)] \
            -foreground [list selected $colors(-selectfg)]
            
        ttk::style configure Treeview.Item \
            -padding {2 0 0 0}
            
        ttk::style configure Heading \
            -background $colors(-bg) \
            -foreground $colors(-fg) \
            -relief flat \
            -padding 2
            
        ttk::style map Heading \
            -background [list active $colors(-hovercolor)]

        # Panedwindow
        ttk::style configure TPanedwindow \
            -background $colors(-bg)
            
        ttk::style configure Sash \
            -sashthickness 4 \
            -gripcount 10
    }
}

# Load the theme
package provide ttk::theme::azure-dark 1.0
ttk::style theme use azure-dark
