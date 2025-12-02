import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { SignedIn, SignedOut, UserButton } from "@clerk/clerk-react";
import { Menu, X, Music, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();
  const isLanding = location.pathname === "/";

  const navItems = [
    { name: "Features", href: "#features" },
    { name: "How It Works", href: "#how-it-works" },
    { name: "About", href: "#about" },
  ];

  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
    e.preventDefault();
    const element = document.querySelector(href);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
      setIsOpen(false);
    }
  };

  return (
    <nav className={cn(
      "fixed top-0 left-0 right-0 z-50 transition-all duration-500",
      "bg-background/80 backdrop-blur-xl border-b border-border/50"
    )}>
      <div className="container mx-auto px-4 lg:px-6">
        <div className="flex items-center justify-between h-20">
          {/* Enhanced Logo */}
          <Link to="/" className="flex items-center space-x-3 group">
            <div className="relative">
              <div className="p-3 rounded-2xl premium-card group-hover:scale-110 transition-all duration-300">
                <Music className="h-7 w-7 text-primary" />
              </div>
              <Sparkles className="absolute -top-1 -right-1 h-4 w-4 text-accent animate-pulse" />
            </div>
            <span className="font-sora font-bold text-2xl text-primary group-hover:text-accent transition-colors">
              ChordAI
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden lg:flex items-center space-x-8">
            {isLanding && navItems.map((item) => (
              <a
                key={item.name}
                href={item.href}
                onClick={(e) => handleNavClick(e, item.href)}
                className="relative text-muted-foreground hover:text-primary transition-all duration-300 font-medium group"
              >
                {item.name}
                <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-primary to-accent group-hover:w-full transition-all duration-300" />
              </a>
            ))}
          </div>

          {/* Desktop Actions */}
          <div className="hidden lg:flex items-center space-x-4">
            <SignedOut>
              <Link to="/login">
                <Button 
                  variant="ghost" 
                  className="text-muted-foreground hover:text-primary hover:bg-primary/10 font-medium px-6 py-2 rounded-xl"
                >
                  Login
                </Button>
              </Link>
              <Link to="/signup">
                <Button className="magnetic-btn px-6 py-2 rounded-xl font-semibold">
                  Get Started
                </Button>
              </Link>
            </SignedOut>
            <SignedIn>
              <Link to="/dashboard">
                <Button 
                  variant="ghost" 
                  className="text-muted-foreground hover:text-primary hover:bg-primary/10 font-medium px-6 py-2 rounded-xl"
                >
                  Dashboard
                </Button>
              </Link>
              <UserButton 
                appearance={{
                  elements: {
                    avatarBox: "w-10 h-10",
                    userButtonPopoverCard: "premium-card"
                  }
                }}
              />
            </SignedIn>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="lg:hidden p-3 rounded-xl hover:bg-muted/50 transition-all duration-300"
          >
            {isOpen ? (
              <X className="h-6 w-6 text-foreground" />
            ) : (
              <Menu className="h-6 w-6 text-foreground" />
            )}
          </button>
        </div>

        {/* Enhanced Mobile Menu */}
        {isOpen && (
          <div className="lg:hidden">
            <div className="premium-card mx-4 mb-4 p-6 animate-slide-up">
              <div className="flex flex-col space-y-6">
                {isLanding && navItems.map((item) => (
                  <a
                    key={item.name}
                    href={item.href}
                    onClick={(e) => handleNavClick(e, item.href)}
                    className="text-muted-foreground hover:text-primary transition-colors duration-300 font-medium py-2"
                  >
                    {item.name}
                  </a>
                ))}
                <div className="flex flex-col space-y-3 pt-6 border-t border-border/50">
                  <SignedOut>
                    <Link to="/login" onClick={() => setIsOpen(false)}>
                      <Button variant="outline" className="w-full justify-center py-3 rounded-xl">
                        Login
                      </Button>
                    </Link>
                    <Link to="/signup" onClick={() => setIsOpen(false)}>
                      <Button className="w-full magnetic-btn py-3 rounded-xl font-semibold">
                        Get Started
                      </Button>
                    </Link>
                  </SignedOut>
                  <SignedIn>
                    <Link to="/dashboard" onClick={() => setIsOpen(false)}>
                      <Button className="w-full magnetic-btn py-3 rounded-xl font-semibold">
                        Dashboard
                      </Button>
                    </Link>
                  </SignedIn>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
