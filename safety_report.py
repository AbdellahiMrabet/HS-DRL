# safety_report.py - Generate safety comparison report

def generate_safety_report(results_dir='results'):
    """Generate a comprehensive safety comparison report"""
    
    loader = ResultsLoader(results_dir)
    data = loader.load_all()
    
    print("\n" + "="*80)
    print("SAFETY COMPARISON REPORT")
    print("="*80)
    
    print("\n📊 **SAFETY METRICS SUMMARY**")
    print("-"*80)
    print(f"{'Metric':<35} {'DRS':<12} {'RLSK':<12} {'EPRS':<12} {'HS-DRL':<12}")
    print("-"*80)
    
    metrics_to_compare = [
        ('Safety Compliance Rate (%)', 'safety_compliance_rate'),
        ('Constraint Violations', 'constraint_violations'),
        ('Node Overload Events', 'node_overloads'),
        ('Avg CPU Margin (%)', 'avg_cpu_margin'),
        ('Projection Rate (%)', 'projection_rate'),
        ('Unsafe Actions Corrected (%)', 'unsafe_correction_rate')
    ]
    
    for metric_name, key in metrics_to_compare:
        row = [metric_name]
        for agent in ['DRS', 'RLSK', 'EPRS', 'HS-DRL']:
            if agent in data and data[agent]:
                last_ep = data[agent][-1]
                value = last_ep.get(key, 0)
                if 'Margin' in metric_name:
                    row.append(f"{value*100:.1f}%")
                elif 'Rate' in metric_name or '%' in metric_name:
                    row.append(f"{value:.1f}%")
                else:
                    row.append(f"{value:.0f}")
            else:
                row.append("N/A")
        print(f"{row[0]:<35} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12}")
    
    print("\n📈 **KEY FINDINGS**")
    print("="*80)
    
    hsdrl_data = data.get('HS-DRL', [])
    if hsdrl_data:
        last_hsdrl = hsdrl_data[-1]
        print(f"\n✓ HS-DRL achieves {last_hsdrl.get('safety_compliance_rate', 0):.1f}% safety compliance")
        print(f"✓ HS-DRL corrects {last_hsdrl.get('unsafe_correction_rate', 0):.1f}% of unsafe actions via projection")
        print(f"✓ HS-DRL maintains {last_hsdrl.get('avg_cpu_margin', 0)*100:.1f}% average CPU safety margin")
    
    # Calculate improvement over best baseline
    baseline_best_safety = 0
    for agent in ['DRS', 'RLSK', 'EPRS']:
        if agent in data and data[agent]:
            safety = data[agent][-1].get('safety_compliance_rate', 0)
            baseline_best_safety = max(baseline_best_safety, safety)
    
    if hsdrl_data:
        hsdrl_safety = hsdrl_data[-1].get('safety_compliance_rate', 0)
        improvement = hsdrl_safety - baseline_best_safety
        print(f"\n🏆 HS-DRL improves safety compliance by {improvement:.1f}% over the best baseline")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    generate_safety_report()
