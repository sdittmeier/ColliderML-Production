import time
import math
from pathlib import Path
import acts
import acts.examples
from acts.examples import Sequencer
from acts.examples.hepmc3 import (
        HepMC3Writer,
        HepMC3Reader,
    )
from acts.examples.pythia8 import Pythia8Generator
import traceback
# Removed pyhepmc, numpy imports as they will be handled by the imported module

from utils.app_logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config


u = acts.UnitConstants

LOG_LEVEL = acts.logging.INFO

def parse_args():
    """Parse command line arguments"""
    parser = create_base_parser("Pythia8 event generation and ACTS merging")
    
    # Generation control
    parser.add_argument(
        "--generate-hard-scatter",
        help="Generate hard scatter events with Pythia8",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--generate-pileup",
        help="Generate pileup events with Pythia8", 
        action="store_true",
        default=None,
    )
    
    # Merging control
    parser.add_argument(
        "--merge",
        help="Merge hard scatter and pileup using ACTS HepMC3Reader",
        action="store_true", 
        default=None,
    )
    
    # File paths (optional overrides)
    parser.add_argument(
        "--hard-scatter-file",
        help="Path to hard scatter file (auto-detect if not specified)",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--pileup-file", 
        help="Path to pileup file (auto-detect if not specified)",
        type=Path,
        default=None,
    )
    
    # Pythia8 settings
    parser.add_argument(
        "--pileup",
        help="Number of pile-up events per hard scatter",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--hard-process",
        help="Pythia8 hard process",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pythia-settings",
        help="Additional Pythia8 settings (comma-separated)",
        type=str,
        default=None,
    )
    
    # Vertex smearing parameters 
    parser.add_argument(
        "--vertex-sigma-xy",
        help="Sigma for vertex smearing in x/y [mm]",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--vertex-sigma-z",
        help="Sigma for vertex smearing in z [mm]",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--vertex-sigma-t",
        help="Sigma for vertex smearing in time [ns]",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--threads",
        help="Number of threads to use",
        type=int,
        default=None,
    )
    return parser.parse_args()

def create_vertex_generator(config, logger):
    """Create ACTS GaussianVertexGenerator from config parameters."""
    sigma_xy = getattr(config, 'vertex_sigma_xy', 0.0) or 0.0
    sigma_z = getattr(config, 'vertex_sigma_z', 0.0) or 0.0
    sigma_t = getattr(config, 'vertex_sigma_t', 0.0) or 0.0
    
    logger.info(f"Vertex smearing: sigma_xy={sigma_xy} mm, sigma_z={sigma_z} mm, sigma_t={sigma_t} ns")
    
    if any(sigma != 0 for sigma in [sigma_xy, sigma_z, sigma_t]):
        return acts.examples.GaussianVertexGenerator(
            stddev=acts.Vector4(
                sigma_xy * u.mm,
                sigma_xy * u.mm, 
                sigma_z * u.mm,
                sigma_t * u.ns
            ),
            mean=acts.Vector4(0, 0, 0, 0),
        )
    else:
        logger.info("No vertex smearing configured")
        return None

def parse_pythia_settings(config, logger):
    """Parse Pythia8 settings from config and hard process."""
    pythia_settings = []
    
    # Add settings from config
    if getattr(config, 'pythia_settings', None):
        if isinstance(config.pythia_settings, str):
            pythia_settings.extend([s.strip() for s in config.pythia_settings.split(',') if s.strip()])
        elif isinstance(config.pythia_settings, list):
            pythia_settings.extend(config.pythia_settings)
        else:
            raise ValueError("pythia_settings must be either a comma-separated string or a list")
    
    # Add hard process
    hard_process = getattr(config, 'hard_process', None)
    if isinstance(hard_process, str) and hard_process.strip():
        pythia_settings.append(f"{hard_process.strip()}=on")

    return pythia_settings if pythia_settings else None

def _build_pythia_event(settings, seed, vertex=None):
    """Create an EventGenerator producing HepMC3 events via Pythia8."""
    rnd = acts.examples.RandomNumbers(seed=seed)
    generator = acts.examples.EventGenerator.Generator(
        multiplicity=acts.examples.FixedMultiplicityGenerator(n=1),
        vertex=vertex
        or acts.examples.GaussianVertexGenerator(
            mean=acts.Vector4(0, 0, 0, 0), stddev=acts.Vector4(0, 0, 0, 0)
        ),
        particles=Pythia8Generator(
            level=LOG_LEVEL,
            settings=settings or [],
        ),
    )
    return acts.examples.EventGenerator(
        level=LOG_LEVEL,
        generators=[generator],
        randomNumbers=rnd,
        outputEvent="pythia8-event",
    )


def generate_hard_scatter(output_dir, config, logger):
    """Generate hard scatter events with Pythia8.
    
    Returns:
        Path: Path to generated hard scatter file
    """
    pythia_settings = parse_pythia_settings(config, logger)
    if not pythia_settings:
        raise ValueError("No hard process configured for hard scatter generation")
    
    logger.info(f"Generating {config.events} hard scatter events")
    
    s = Sequencer(
        numThreads=config.threads if config.threads is not None else 1,
        events=config.events
    )
    s.config.logLevel = LOG_LEVEL
    seed = config.seed or int(time.time())
    output_path = output_dir / "events_signal.hepmc3"
    
    s.addReader(_build_pythia_event(pythia_settings, seed, vertex=None))
    
    s.addWriter(
        HepMC3Writer(
            LOG_LEVEL,
            inputEvent="pythia8-event",
            outputPath=output_path,
        )
    )
    
    logger.info(f"Writing hard scatter events to {output_path}")
    s.run()
    logger.info("Hard scatter generation completed")
    
    return output_path

def generate_pileup(output_dir, config, logger):
    """Generate individual pileup events for ACTS merging.
    
    Returns:
        Path: Path to generated pileup file
    """
    # Calculate total pileup events to generate
    pileup_mu = getattr(config, 'pileup', 1)
    if getattr(config, 'poisson_sample', False):
        expected = config.events * pileup_mu
        z_sigma = getattr(config, 'poisson_buffer_sigma', 5.0)
        total_pileup_events = int(math.ceil(expected + z_sigma * math.sqrt(expected)))
        logger.info(
            "Generating %d individual pileup events using Poisson buffer (E=%d, mu=%d, z=%.2f)"
            % (total_pileup_events, config.events, pileup_mu, z_sigma)
        )
    else:
        total_pileup_events = config.events * pileup_mu
        logger.info(
            f"Generating {total_pileup_events} individual pileup events "
            f"({config.events} signal events × {pileup_mu} pileup per signal)"
        )
    
    s = Sequencer(
        numThreads=config.threads if config.threads is not None else 1,
        events=total_pileup_events
    )
    s.config.logLevel = LOG_LEVEL
    seed = (config.seed or int(time.time())) + 1000
    output_path = output_dir / "events_pileup.hepmc3"
    
    pileup_settings = ["SoftQCD:all = on"]
    s.addReader(_build_pythia_event(pileup_settings, seed, vertex=None))
    
    s.addWriter(
        HepMC3Writer(
            LOG_LEVEL,
            inputEvent="pythia8-event",
            outputPath=output_path,
        )
    )
    
    logger.info(f"Writing individual pileup events to {output_path}")
    s.run()
    logger.info("Pileup generation completed")
    
    return output_path

def find_hard_scatter_file(output_dir, config, explicit_path=None):
    """Find hard scatter file with smart detection."""
    explicit_path = Path(explicit_path) if explicit_path else None
    if explicit_path and explicit_path.exists():
        return explicit_path
    
    # Check config for hard scatter file path
    config_path = getattr(config, 'hard_scatter_file', None)
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            return config_path
    
    # Auto-detect in output directory
    candidates = [
        output_dir / "events_signal.hepmc3",  # From Pythia8 generation
        output_dir / "events.hepmc3",         # From MadGraph with splitting
        output_dir / "events.hepmc",          # From MadGraph without splitting
        output_dir / "events.hepmc.gz",      # From MadGraph without splitting
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    raise FileNotFoundError(f"Hard scatter file not found. Looked for: {[str(c) for c in candidates]}")

def find_pileup_file(output_dir, config, explicit_path=None):
    """Find pileup file with smart detection."""
    explicit_path = Path(explicit_path) if explicit_path else None
    if explicit_path and explicit_path.exists():
        return explicit_path
    
    # Check config for pileup file path
    config_path = getattr(config, 'pileup_file', None)
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            return config_path
    
    # Auto-detect in output directory
    pileup_path = output_dir / "events_pileup.hepmc3"
    if pileup_path.exists():
        return pileup_path
    
    raise FileNotFoundError(f"Pileup file not found: {pileup_path}")

def merge_events(hard_scatter_file, pileup_file, output_dir, config, logger):
    """Merge hard scatter and pileup events using ACTS HepMC3Reader.
    
    Returns:
        Path: Path to merged output file
    """
    pileup_multiplicity = getattr(config, 'pileup', 1)
    
    logger.info(f"ACTS merging:")
    logger.info(f"  Hard scatter: {hard_scatter_file}")
    logger.info(f"  Pileup: {pileup_file}")
    logger.info(f"  Pileup multiplicity: {pileup_multiplicity}")
    
    # Create vertex generator
    vtxGen = create_vertex_generator(config, logger)
    
    # Create sequencer
    s = acts.examples.Sequencer(
        numThreads=config.threads if config.threads is not None else 1,
        events=config.events
    )
    s.config.logLevel = LOG_LEVEL
    
    # Random number generator
    rng = acts.examples.RandomNumbers(seed=config.seed or 42)
    
    # ACTS HepMC3Reader with merging
    # Build inputs using HepMC3Reader.Input API
    if getattr(config, 'poisson_sample', False):
        # Use Poisson sampling for pileup
        inputs = [
            HepMC3Reader.Input.Fixed(hard_scatter_file, 1),
            HepMC3Reader.Input.Poisson(pileup_file, float(pileup_multiplicity))
        ]
    else:
        # Use fixed multiplicity (backward compatible)
        inputs = [
            HepMC3Reader.Input.Fixed(hard_scatter_file, 1),
            HepMC3Reader.Input.Fixed(pileup_file, max(1, int(pileup_multiplicity)))
        ]

    s.addReader(
        HepMC3Reader(
            inputs=inputs,
            level=LOG_LEVEL,
            outputEvent="merged_events",
            randomNumbers=rng,
            vertexGenerator=vtxGen,
            numEvents=config.events,
        )
    )
    
    # Output merged file
    merged_path = output_dir / "merged_events.hepmc3"
    s.addWriter(
        HepMC3Writer(
            inputEvent="merged_events",
            outputPath=merged_path,
            level=LOG_LEVEL,
            writeEventsInOrder=False,
        )
    )
    
    logger.info(f"Running ACTS merge to: {merged_path}")
    s.run()
    logger.info("ACTS merging completed")
    
    return merged_path

def determine_workflow(config, logger):
    """Determine what operations to perform based on config."""
    # Simplest logic: check config directly
    should_generate_hard_scatter = bool(getattr(config, 'hard_process', None))
    should_generate_pileup = getattr(config, 'pileup', 0) > 0
    should_merge = getattr(config, 'merge', False)
    
    logger.info("Workflow determined:")
    logger.info(f"  → Generate hard scatter: {should_generate_hard_scatter}")
    logger.info(f"  → Generate pileup: {should_generate_pileup}")
    logger.info(f"  → Merge events: {should_merge}")
    
    return should_generate_hard_scatter, should_generate_pileup, should_merge

def run_workflow(output_dir, config, logger):
    """Execute the complete workflow based on configuration.
    
    Returns:
        Path: Path to final output file
    """
    should_generate_hard_scatter, should_generate_pileup, should_merge = determine_workflow(config, logger)
    
    # Generation Phase
    hard_scatter_file = None
    pileup_file = None
    
    if should_generate_hard_scatter:
        logger.info("=== GENERATION PHASE: Hard Scatter ===")
        hard_scatter_file = generate_hard_scatter(output_dir, config, logger)
    
    if should_generate_pileup:
        logger.info("=== GENERATION PHASE: Pileup ===")
        pileup_file = generate_pileup(output_dir, config, logger)
    
    # Merging Phase  
    if should_merge:
        logger.info("=== MERGING PHASE ===")
        
        # Find files if not generated
        if not hard_scatter_file:
            hard_scatter_file = find_hard_scatter_file(
                output_dir, config, 
                getattr(config, 'hard_scatter_file', None)
            )
            logger.info(f"Found hard scatter file: {hard_scatter_file}")
        
        if not pileup_file:
            pileup_file = find_pileup_file(
                output_dir, config,
                getattr(config, 'pileup_file', None)
            )
            logger.info(f"Found pileup file: {pileup_file}")
        
        return merge_events(hard_scatter_file, pileup_file, output_dir, config, logger)
    
    # Return the last generated file if no merging
    if hard_scatter_file:
        return hard_scatter_file
    elif pileup_file:
        return pileup_file
    else:
        raise ValueError("No files generated and no merging performed")

def main():
    try:
        # Parse arguments and load config
        args = parse_args()
        config = load_config(args)
        logger = setup_logging("Pythia8ACTS")
        
        # Create output directory
        output_dir = Path(args.output)
        if args.output_subdir:
            output_dir = output_dir / args.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=== Pythia8 + ACTS Workflow ===")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Events: {config.events}")
        
        # Initialize timing
        timer = TimingRecorder(output_dir)
        
        # Run workflow
        with timer.record("Pythia8 + ACTS Workflow"):
            final_output = run_workflow(output_dir, config, logger)
        
        # Write timing report
        timer.write_report()
        
        logger.info("=== Workflow Completed Successfully ===")
        logger.info(f"Final output: {final_output}")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
