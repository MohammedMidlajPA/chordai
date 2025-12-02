import { Upload, Cpu, Music2, Sparkles, ArrowDown } from "lucide-react";
import { Link } from "react-router-dom";

const HowItWorks = () => {
  const steps = [
    {
      icon: Upload,
      step: "01",
      title: "Upload Your Track",
      description: "Drag and drop any audio file or record directly in your browser. We support MP3, WAV, FLAC and more.",
      color: "text-primary"
    },
    {
      icon: Cpu,
      step: "02", 
      title: "Audio Analysis",
      description: "Our algorithms process your audio to detect pitch patterns and identify chord progressions in real-time.",
      color: "text-accent"
    },
    {
      icon: Music2,
      step: "03",
      title: "Get Guitar Chords",
      description: "View your chord progression with guitar fingering diagrams. Play along and learn the song instantly.",
      color: "text-primary"
    }
  ];

  return (
    <section id="how-it-works" className="py-16 sm:py-24 lg:py-32 hero-mesh relative overflow-hidden">
      {/* Enhanced Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/6 w-64 sm:w-80 lg:w-96 h-64 sm:h-80 lg:h-96 rounded-full bg-primary/8 blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/6 w-56 sm:w-72 lg:w-80 h-56 sm:h-72 lg:h-80 rounded-full bg-accent/8 blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-40 sm:w-56 lg:w-72 h-40 sm:h-56 lg:h-72 rounded-full bg-gradient-to-r from-primary/5 to-accent/5 blur-2xl animate-pulse" style={{ animationDelay: '4s' }} />
      </div>

      <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Enhanced Header */}
        <div className="text-center mb-12 sm:mb-16 lg:mb-20 animate-slide-up opacity-0">
          <div className="inline-flex items-center space-x-2 sm:space-x-3 premium-card px-4 sm:px-6 py-2 sm:py-3 mb-6 sm:mb-8 text-xs sm:text-sm backdrop-blur-sm">
            <Sparkles className="h-4 w-4 sm:h-5 sm:w-5 text-accent pulse-glow" />
            <span className="text-foreground font-medium">Simple Process</span>
            <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 rounded-full bg-accent animate-pulse" />
          </div>
          
          <h2 className="font-sora font-bold text-3xl sm:text-4xl md:text-5xl lg:text-6xl mb-4 sm:mb-6 text-foreground leading-tight tracking-tight">
            <span className="text-transparent bg-gradient-to-r from-accent via-primary to-accent bg-clip-text bg-300 bg-pos-0 hover:bg-pos-100 transition-all duration-2000">
              How It Works
            </span>
          </h2>
          <p className="text-muted-foreground text-base sm:text-lg lg:text-xl max-w-2xl lg:max-w-3xl mx-auto leading-relaxed">
            Get from song to guitar chords in three simple steps. Upload, analyze, and start playing along.
          </p>
        </div>

        {/* Enhanced Steps */}
        <div className="max-w-5xl mx-auto">
          {steps.map((step, index) => (
            <div key={index} className="relative mb-12 sm:mb-16 lg:mb-20 last:mb-0">
              <div className="flex flex-col lg:flex-row items-center gap-8 sm:gap-10 lg:gap-16 animate-slide-up opacity-0" style={{ animationDelay: `${index * 0.2 + 0.3}s` }}>
                {/* Enhanced Step Icon & Number */}
                <div className="flex-shrink-0 relative order-1 lg:order-none">
                  <div className="premium-card p-6 sm:p-8 lg:p-10 group hover:scale-105 transition-all duration-500 cursor-pointer">
                    <div className="relative">
                      <div className="w-20 h-20 sm:w-24 sm:h-24 lg:w-28 lg:h-28 rounded-2xl sm:rounded-3xl bg-gradient-to-br from-primary/20 via-accent/15 to-primary/20 flex items-center justify-center mb-4 sm:mb-6 group-hover:scale-110 group-hover:rotate-3 transition-all duration-500 relative overflow-hidden">
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 transform -translate-x-full group-hover:translate-x-full" />
                        <step.icon className={`h-10 w-10 sm:h-12 sm:w-12 lg:h-14 lg:w-14 ${step.color} relative z-10`} />
                      </div>
                      <div className="absolute -top-2 -right-2 sm:-top-3 sm:-right-3 w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-gradient-to-r from-primary to-accent text-background font-bold text-xs sm:text-sm flex items-center justify-center shadow-lg animate-pulse">
                        {step.step}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Enhanced Step Content */}
                <div className="flex-1 text-center lg:text-left order-2 lg:order-none">
                  <h3 className="font-sora font-semibold text-2xl sm:text-3xl lg:text-4xl mb-4 sm:mb-6 text-foreground leading-tight">
                    {step.title}
                  </h3>
                  <p className="text-muted-foreground text-base sm:text-lg lg:text-xl leading-relaxed max-w-xl lg:max-w-2xl mx-auto lg:mx-0">
                    {step.description}
                  </p>
                </div>
              </div>

              {/* Enhanced Connector Arrow */}
              {index < steps.length - 1 && (
                <div className="flex justify-center my-8 sm:my-12 lg:my-16 order-3 lg:order-none">
                  <div className="premium-card p-3 sm:p-4 rounded-full animate-bounce hover:scale-110 transition-transform duration-300 cursor-pointer group">
                    <ArrowDown className="h-5 w-5 sm:h-6 sm:w-6 text-primary group-hover:text-accent transition-colors duration-300" />
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Call to Action */}
        <div className="text-center mt-16 sm:mt-20 lg:mt-24 animate-slide-up opacity-0" style={{ animationDelay: '1s' }}>
          <div className="premium-card p-6 sm:p-8 lg:p-10 max-w-2xl mx-auto">
            <h3 className="font-sora font-semibold text-xl sm:text-2xl lg:text-3xl mb-4 text-foreground">
              Ready to Get Started?
            </h3>
            <p className="text-muted-foreground text-sm sm:text-base lg:text-lg mb-6">
              Start detecting chords from your favorite songs today. No music theory knowledge required.
            </p>
            <Link 
              to="/signup" 
              className="inline-flex items-center justify-center px-6 sm:px-8 py-3 sm:py-4 bg-gradient-to-r from-primary to-accent text-background font-semibold rounded-xl sm:rounded-2xl hover:scale-105 transition-all duration-300 hover:shadow-[0_0_30px_rgba(108,99,255,0.4)]"
            >
              Start Analyzing Music
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
