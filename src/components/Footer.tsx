import { Link } from "react-router-dom";
import { Music, Twitter, Github, Mail } from "lucide-react";

const Footer = () => {
  const footerLinks = {
    Product: [
      { name: "Features", href: "#features" },
      { name: "How It Works", href: "#how-it-works" },
      { name: "Pricing", href: "#" },
      { name: "API", href: "#" }
    ],
    Company: [
      { name: "About", href: "#about" },
      { name: "Blog", href: "#" },
      { name: "Careers", href: "#" },
      { name: "Contact", href: "#" }
    ],
    Support: [
      { name: "Help Center", href: "#" },
      { name: "Community", href: "#" },
      { name: "Privacy", href: "#" },
      { name: "Terms", href: "#" }
    ]
  };

  const socialLinks = [
    { icon: Twitter, href: "#", label: "Twitter" },
    { icon: Github, href: "#", label: "GitHub" },
    { icon: Mail, href: "#", label: "Email" }
  ];

  return (
    <footer className="bg-secondary/95 backdrop-blur-sm text-white relative overflow-hidden">
      {/* Background Enhancement */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-64 h-64 rounded-full bg-primary/5 blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-48 h-48 rounded-full bg-accent/5 blur-2xl" />
      </div>

      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-16 lg:py-20 relative z-10">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-8 lg:gap-12 mb-12">
          {/* Enhanced Brand */}
          <div className="sm:col-span-2 lg:col-span-2 text-center sm:text-left">
            <Link to="/" className="inline-flex items-center space-x-3 mb-6 group">
              <div className="p-3 rounded-xl bg-gradient-to-r from-primary/20 to-accent/20 group-hover:scale-110 transition-transform duration-300">
                <Music className="h-6 w-6 sm:h-7 sm:w-7 text-primary" />
              </div>
              <span className="font-sora font-bold text-xl sm:text-2xl text-primary">ChordAI</span>
            </Link>
            <p className="text-white/80 mb-6 sm:mb-8 max-w-md mx-auto sm:mx-0 text-sm sm:text-base leading-relaxed">
              Transform any song into chords instantly with AI-powered analysis. 
              Perfect for musicians, producers, and music enthusiasts worldwide.
            </p>
            <div className="flex justify-center sm:justify-start space-x-3 sm:space-x-4">
              {socialLinks.map((social, index) => (
                <a 
                  key={index}
                  href={social.href}
                  className="w-10 h-10 sm:w-12 sm:h-12 rounded-lg bg-white/10 hover:bg-primary/20 flex items-center justify-center transition-all duration-300 hover:scale-110 hover:shadow-lg group"
                  aria-label={social.label}
                >
                  <social.icon className="h-4 w-4 sm:h-5 sm:w-5 group-hover:text-primary transition-colors duration-300" />
                </a>
              ))}
            </div>
          </div>

          {/* Enhanced Links */}
          {Object.entries(footerLinks).map(([category, links]) => (
            <div key={category} className="text-center sm:text-left">
              <h3 className="font-sora font-semibold mb-4 sm:mb-6 text-sm sm:text-base text-white">{category}</h3>
              <ul className="space-y-3 sm:space-y-4">
                {links.map((link, index) => (
                  <li key={index}>
                    <a 
                      href={link.href}
                      className="text-white/70 hover:text-primary transition-all duration-300 text-sm sm:text-base hover:translate-x-1 inline-block"
                    >
                      {link.name}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Enhanced Bottom */}
        <div className="pt-6 sm:pt-8 border-t border-white/10 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-white/60 text-xs sm:text-sm text-center sm:text-left">
            © 2024 ChordAI. All rights reserved.
          </p>
          <div className="flex items-center space-x-2 text-white/60 text-xs sm:text-sm">
            <span>Made with</span>
            <span className="text-red-400 animate-pulse">❤️</span>
            <span>for musicians worldwide</span>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;