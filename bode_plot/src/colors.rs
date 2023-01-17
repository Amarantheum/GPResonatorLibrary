use plotters::define_color;
use plotters::prelude::*;

// Macro for allowing dynamic creation of doc attributes.
// Taken from https://stackoverflow.com/questions/60905060/prevent-line-break-in-doc-test
macro_rules! doc {
    {
        $(#[$m:meta])*
        $(
            [$doc:expr]
            $(#[$n:meta])*
        )*
        @ $thing:item
    } => {
        $(#[$m])*
        $(
            #[doc = $doc]
            $(#[$n])*
        )*
        $thing
    }
}
define_color!(LIGHT_BLUE, 0x1d, 0xa5, 0xe2, "Light Blue");